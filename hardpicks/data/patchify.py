import itertools
import typing

import numpy as np
import PIL.Image
import torch

import hardpicks.utils.patch_utils as patches


def flex_crop(
    image: typing.Union[np.ndarray, torch.Tensor],
    patch: patches.PatchCoord,
    padding_val: typing.Union[int, float] = 0,
    force_copy: bool = False,
) -> typing.Union[np.ndarray, torch.Tensor]:
    """Flexibly crops a region from within an OpenCV or PyTorch image, using padding if needed.

    If the image is provided as a numpy array with two or three dimensions, it will be cropped under
    the assumption that it is an OpenCV-like image with a number of channels of 1 to 4 and that the
    channel dimension is the last in the array shape (e.g. H x W x C), resulting in patches with
    a similar dimension ordering. Otherwise, the image will be cropped under the assumption that we
    are doing a 2D spatial crop and that the spatial dimensions are the last dimensions in the
    array shape (e.g. for a 4-dim tensor B x C x H x W and a 2D crop size, the output will be
    B x C x [patch.shape]).

    Args:
        image: the image to crop (provided as a numpy array or torch tensor).
        patch: a patch coordinates object describing the area to crop from the image.
        padding_val: border value to use when padding the image.
        force_copy: defines whether to force a copy of the target image region even when it can be
            avoided.

    Returns:
        The cropped image of the same object type as the input image, with the same dim arrangement.
    """
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise TypeError("expected input image to be numpy array or torch tensor")

    if isinstance(image, np.ndarray) and image.ndim in [2, 3] and patch.ndim == 2:
        # special handling for OpenCV-like arrays (where the channel is the last dimension)
        assert image.ndim == 2 or image.shape[2] in [1, 2, 3, 4], \
            "cannot handle channel counts outside [1, 2, 3, 4] with opencv crop!"
        image_region = patches.PatchCoord((0, 0), shape=image.shape[:2])
        if patch not in image_region:
            # special handling for crop coordinates falling outside the image bounds...
            crop_out_shape = patch.shape if image.ndim == 2 else (patch.shape, image.shape[2])
            inters_coord = patch.intersection(image_region)
            # this forces an allocation, as we cannot get a view with the required padding
            crop_out = np.full(shape=crop_out_shape, fill_value=padding_val, dtype=image.dtype)
            if inters_coord is not None:  # if the intersection contains any pixels at all, copy...
                offset = [inters_coord.tl[0] - patch.tl[0], inters_coord.tl[1] - patch.tl[1]]
                crop_out[
                    offset[0]:(offset[0] + inters_coord.shape[0]),
                    offset[1]:(offset[1] + inters_coord.shape[1]),
                    ...
                ] = image[
                    inters_coord.tl[0]:inters_coord.br[0],
                    inters_coord.tl[1]:inters_coord.br[1],
                    ...
                ]
            return crop_out
        # if all crop coordinates are in-bounds, we can get a view directly from the image
        crop_view = image[patch.tl[0]:patch.br[0], patch.tl[1]:patch.br[1], ...]
        if not force_copy:
            return crop_view
        return np.copy(crop_view)

    # regular handling (we crop along the spatial dimensions located at the end of the array)
    assert patch.ndim <= image.ndim, \
        "patch dim count should be equal to or lower than image dimension count!"
    image_region = patches.PatchCoord([0] * patch.ndim, shape=image.shape[-patch.ndim:])
    crop_out_shape = tuple(image.shape[:-patch.ndim]) + patch.shape

    # first check: figure out if there is anything to crop at all
    inters_coord = patch.intersection(image_region)
    if inters_coord is None or inters_coord.is_empty:
        # if not, we can return right away without any allocation (out shape will have a zero-dim)
        if isinstance(image, torch.Tensor):
            return torch.empty(crop_out_shape, dtype=image.dtype, device=image.device)
        else:
            return np.empty(crop_out_shape, dtype=image.dtype)

    # if there is an intersection, figure out if it's totally inside the image or not
    if patch not in image_region:
        # ...it's not totally in the image, so we'll have to allocate + fill
        if isinstance(image, torch.Tensor):
            crop_out = \
                torch.full(crop_out_shape, padding_val, dtype=image.dtype, device=image.device)
        else:
            crop_out = np.full(crop_out_shape, padding_val, dtype=image.dtype)
        offset = [inters_coord.tl[d] - patch.tl[d] for d in patch.dimrange]
        crop_inner_slice = tuple([slice(None)] * (image.ndim - patch.ndim) + [
            slice(offset[d], offset[d] + inters_coord.shape[d])
            for d in patch.dimrange
        ])
        crop_outer_slice = tuple([slice(None)] * (image.ndim - patch.ndim)) + inters_coord.slice
        crop_out[crop_inner_slice] = image[crop_outer_slice]
        return crop_out

    # if we get here, there is an intersection without any out-of-bounds element lookup
    crop_outer_slice = tuple([slice(None)] * (image.ndim - patch.ndim)) + inters_coord.slice
    crop_view = image[crop_outer_slice]
    if not force_copy:
        return crop_view
    elif isinstance(image, np.ndarray):
        return np.copy(crop_view)
    else:
        return crop_view.clone()  # note: this will not detach the tensor, just make a copy!


class Patchify:
    """Returns an array of patches cut out from a given image based a regular grid.

    This operation can perform patching given an optional mask with a target intersection over union
    (IoU) score, and with an optional overlap between patches. If a mask is used, the first patch
    position is tested exhaustively by iterating over all spatial coordinates starting from the
    top-left corner of the image, looking for a starting location with the given IoU threshold.
    Otherwise, the first patch position is set as the origin (top left corner) of the image. Then,
    all other patches are found by offsetting from these initial coordinates along each spatial axis
    of the image, and while checking the IoU threshold with the mask (if needed).

    To see how the patches are arranged in terms of output array shapes, see the documentation of
    the `__call__` function.

    Attributes:
        patch_shape: size of the output patches, provided as an N-dim tuple or list (for N-dim
            patches). If integer values are used, the size is assumed to be absolute. If floating
            point values are used (i.e. in [0,1]), the output size is assumed to be relative to
            the original image size, and will be determined at execution time for each image.
        patch_overlap: overlap allowed between two neighboring patches; should be a ratio in [0,1],
            where 0 means that patches will not overlap at all (they will only be co-located).
        min_mask_iou: minimum mask intersection over union (IoU) required for accepting a patch
            (in [0,1]). Has no effect if a mask is not used.
        offset_overlap: specifies whether the 'overlap' of patches should be started outside the
            image bounds or not. For example, for a 2D image being cut with 2D patches of size 10x10
            and using a 50% overlap ratio, if this flag is set as false, the first patch will have
            its origin (top-left coordinates) at (0, 0). If this flag is set as true, with the same
            parameters, the patch's origin (top-left coordinates) will be (-5, -5).
        padding_val: padding value to use when the image is too small for the required crop size, or
            when using the above `offset_overlap` option as true.
        make_contiguous: defines whether the transform op call will return a contiguous memory block
            (under a new allocation) or a list of 'views' of the patches in the original tensor (if
            possible).
        jittered_subcrop_shape: size of the 'jittered' (randomly positioned) subcrops to extract
            from the initial patches, if any. If no tuple is provided here, no subcropping will
            occur. If a tuple is provided here, it must be the same length as `patch_shape`, and it
            must contain either absolute values smaller than the initial patch size, or a relative
            value below 1. Subcrops will be generated using this shape and padded back to the
            original patch size.
    """

    def __init__(
        self,
        patch_shape: typing.Tuple[typing.Union[int, float], ...],
        patch_overlap: float = 0.0,  # should be in [0, 1[
        min_mask_iou: float = 1.0,  # should be in [0, 1]
        offset_overlap: bool = False,
        padding_val: typing.Union[int, float] = 0,
        make_contiguous: bool = True,  # if false, the transform will return 'views' if possible
        jittered_subcrop_shape: typing.Optional[typing.Tuple[typing.Union[int, float], ...]] = None,
    ):
        """Validates and initializes patching parameters.

        Args:
            patch_shape: size of the output patches, provided as an N-dim tuple or list (for N-dim
                patches). If integer values are used, the size is assumed to be absolute. If floating
                point values are used (i.e. in [0,1]), the output size is assumed to be relative to
                the original image size, and will be determined at execution time for each image.
            patch_overlap: overlap allowed between two neighboring patches; should be a ratio in [0,1],
                where 0 means that patches will not overlap at all (they will only be co-located).
            min_mask_iou: minimum mask intersection over union (IoU) required for accepting a patch
                (in [0,1]). Has no effect if a mask is not used.
            offset_overlap: specifies whether the 'overlap' of patches should be started outside the
                image bounds or not. For example, for a 2D image being cut with 2D patches of size 10x10
                and using a 50% overlap ratio, if this flag is set as false, the first patch will have
                its origin (top-left coordinates) at (0, 0). If this flag is set as true, with the same
                parameters, the patch's origin (top-left coordinates) will be (-5, -5).
            padding_val: padding value to use when the image is too small for the required crop size, or
                when using the above `offset_overlap` option as true.
            make_contiguous: defines whether the transform op call will return a contiguous memory block
                (under a new allocation) or a list of 'views' of the patches in the original tensor (if
                possible).
            jittered_subcrop_shape: size of the 'jittered' (randomly positioned) subcrops to extract
                from the initial patches, if any. If no tuple is provided here, no subcropping will
                occur. If a tuple is provided here, it must be the same length as `patch_shape`, and it
                must contain either absolute values smaller than the initial patch size, or a relative
                value below 1. Subcrops will be generated using this shape and padded back to the
                original patch size.
        """
        assert isinstance(patch_shape, (tuple, list)), \
            "unexpected patch size type (need tuple or list of int/float values)"
        assert len(patch_shape) > 0, "patch size cannot be a 0-dim vector"
        assert (
            all([isinstance(s, int) and s > 0 for s in patch_shape])
            or all([isinstance(s, float) and 0 < s <= 1 for s in patch_shape])
        ), "expected patch size elements to be the same type (int or float) and strictly positive"
        self.patch_shape = tuple(patch_shape)
        assert isinstance(patch_overlap, (int, float)) and 0 <= patch_overlap < 1, \
            "invalid patch overlap, should be float in [0,1["
        self.patch_overlap = float(patch_overlap)
        assert isinstance(min_mask_iou, (int, float)) and 0 <= min_mask_iou <= 1, \
            "invalid minimum mask IoU score, should be float in [0,1]"
        self.min_mask_iou = float(min_mask_iou)
        self.offset_overlap = offset_overlap
        self.padding_val = padding_val
        self.make_contiguous = make_contiguous
        if jittered_subcrop_shape is not None:
            assert len(jittered_subcrop_shape) == len(patch_shape), \
                "jittered subcrop patch shape tuple must be same dim count as original patch shape"
            assert (
                all([isinstance(s, int) and s > 0 for s in jittered_subcrop_shape])
                or all([isinstance(s, float) and 0 < s < 1 for s in jittered_subcrop_shape])
            ), "invalid subcrop shape values (need int/float and smaller than patch shape values)"
            jittered_subcrop_shape = tuple(jittered_subcrop_shape)
        self.jittered_subcrop_shape = jittered_subcrop_shape
        self._patch_coords_cache = {}  # cache used to speed up patch creation when not using a mask

    def _get_abs_patch_shape_from_image_shape(
        self,
        image_shape: typing.Tuple[int, ...],
    ) -> typing.Tuple[int, ...]:
        """Returns the (absolute) patch shape to use based on the given image shape."""
        # note: this returns the patch shape before any jittered-crop operation is potentially applied!
        if isinstance(self.patch_shape[0], float):  # if we're using relative coordinates...
            assert len(image_shape) >= len(self.patch_shape), "image dim count too low for patch dim count"
            image_spatial_shape = image_shape[-len(self.patch_shape):]
            return tuple([
                max(int(round(image_spatial_shape[d] * self.patch_shape[d])), 1)
                for d in range(len(self.patch_shape))
            ])
        return self.patch_shape  # otherwise, return the absolute shape as-is

    def _get_abs_subcrop_shape_from_patch_shape(
        self,
        patch_shape: typing.Tuple[int, ...],  # note: NO FLOATS HERE!
    ) -> typing.Tuple[int, ...]:
        """Returns the (absolute) subcrop patch shape to use based on the given patch shape (if needed)."""
        if self.jittered_subcrop_shape is None:
            return patch_shape  # no cropping to be done, keep the same shape!
        assert len(patch_shape) == len(self.jittered_subcrop_shape), "unexpected patch shape dim count"
        if isinstance(self.jittered_subcrop_shape[0], float):  # if we're using a relative size...
            return tuple([
                max(int(round(patch_shape[d] * self.jittered_subcrop_shape[d])), 1)
                for d in range(len(self.patch_shape))
            ])
        return self.jittered_subcrop_shape  # otherwise, return the absolute shape as-is

    def _get_offset_steps_from_abs_patch_shape(
        self,
        patch_shape: typing.Tuple[int, ...],
    ) -> typing.Tuple[int, ...]:
        """Returns the (absolute) offset steps to use based on the given patch shape + overlap."""
        assert len(patch_shape) > 0 and all([s > 0 for s in patch_shape]), "bad input patch shape"
        return tuple([
            max(patch_shape[d] - int(round(patch_shape[d] * self.patch_overlap)), 1)
            for d in range(len(patch_shape))
        ])

    def _prepare_inputs(
        self,
        image: typing.Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        mask: typing.Optional[typing.Union[PIL.Image.Image, np.ndarray, torch.Tensor]] = None,
    ) -> typing.Tuple[
        typing.Union[np.ndarray, torch.Tensor],
        typing.Union[np.ndarray, torch.Tensor]
    ]:
        """Returns a (possibly) converted version of the input image to process.

        If the input is an OpenCV-like array, its channel ordering will first be flipped around
        to the torch-like ordering (i.e. `H x W x C` will become `C x H x W`). If a mask is
        provided, it must only have the spatial dimensions of the input image (and no channels!).
        """
        assert isinstance(image, (PIL.Image.Image, np.ndarray, torch.Tensor))
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image)
            assert image.ndim in [2, 3]
            if image.ndim == 3:
                assert image.shape[2] in [1, 2, 3, 4]
                image = np.transpose(image, (2, 0, 1))
        elif isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] in [1, 2, 3, 4]:
            image = np.transpose(image, (2, 0, 1))
        if mask is not None:
            assert isinstance(mask, (PIL.Image.Image, np.ndarray, torch.Tensor))
            if isinstance(mask, PIL.Image.Image):
                mask = np.asarray(mask)
                assert mask.ndim in [2, 3]
                if mask.ndim == 3:
                    assert mask.shape[2] == 1
                    mask = np.squeeze(mask, axis=2)
            elif isinstance(mask, np.ndarray) and mask.ndim == 3:
                assert mask.shape[2] == 1
                mask = np.squeeze(mask, axis=2)
            assert image.shape[-len(self.patch_shape):] == mask.shape
        return image, mask

    def __call__(
        self,
        image: typing.Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        mask: typing.Optional[typing.Union[PIL.Image.Image, np.ndarray, torch.Tensor]] = None,
    ) -> typing.Union[
        typing.Union[np.ndarray, torch.Tensor],
        typing.List[typing.Union[np.ndarray, torch.Tensor]],
    ]:
        """Extracts and returns an array of patches taken from the given image.

        If `make_contiguous` is true and no mask is used, the output will be a contiguous array
        of patches for which the shape will be a concatenation of the grid dimensions (which are
        the number of patches along each spatial axis of the array), the array's non-spatial
        dimensions (including batch and channel), and the patch dimensions. For example, if we
        are splitting an array that is shaped as `B x C x H x W` using 2D patches of size K x K
        without overlap and without offsetting, the output shape is `H/K x W/K x B x C x K x K`.

        If `make_contiguous` is true and a mask is used, the output will be a contiguous array of
        patches for which the shape will be the flattened version of the above. In other words,
        it will have one new dimension (at the first position) that corresponds to the total number
        of patches that were extracted.

        If `make_contiguous` is false, the output will be a list of patches that may be views into
        the original (input) tensor, possibly saving some time spent doing memory copies.

        To get the coordinates of the patches that were extracted in the last two cases, refer to
        the `get_patch_coords` function.

        Note that OpenCV-like images given in `H x W x C` format will automatically be converted
        to `C x H x W` under the hood; to avoid this, transpose the channels before and/or after
        calling this function.

        Args:
            image: the image to cut into patches. This image can have more dimensions than the
                number of dimensions provided in the patch size given in the constructor; these will
                remain untouched and be returned in the output array.
            mask: the mask (ROI) indicating the area where to extract the patches (can be ``None``);
                should only cover the spatial dimensions of the input image.

        Returns:
            A contiguous array of patches (if `make_contiguous` is true) or a list patch arrays.
        """
        # prepare the tensors + coords of patches we'll be cutting into them
        image, mask = self._prepare_inputs(image, mask)
        patch_coords_map = self.get_patch_coords(image, mask)  # pre-jittered-subcrop coords!
        patch_shape = self._get_abs_patch_shape_from_image_shape(image.shape)
        assert all([p.shape == patch_shape for p in patch_coords_map.values()])
        subcrop_patch_shape = self._get_abs_subcrop_shape_from_patch_shape(patch_shape)
        assert all([c <= p for c, p in zip(subcrop_patch_shape, patch_shape)])
        subcrop_shape_diffs = tuple([p - c for c, p in zip(subcrop_patch_shape, patch_shape)])
        subcrop_offsets = tuple([list(range(s + 1)) for s in subcrop_shape_diffs])

        # assemble the list of patches right away (these should be views in the orig data)
        output = []
        for patch_loc, patch_coords in patch_coords_map.items():
            if subcrop_patch_shape != patch_coords.shape:
                # if we need to take a jittered-and-padded subcrop of the patch...
                subcrop = flex_crop(  # this call takes a subview without copy
                    image=image,
                    patch=patches.PatchCoord(
                        top_left=[
                            patch_coords.top_left[d] + np.random.choice(subcrop_offsets[d])
                            for d in range(len(patch_shape))
                        ],
                        shape=subcrop_patch_shape,
                    ),
                    padding_val=self.padding_val,
                    force_copy=False,
                )
                padded_subcrop = flex_crop(  # this call reallocates the view into a padded array
                    image=subcrop,
                    patch=patches.PatchCoord(
                        top_left=[-(shape_diff // 2) for shape_diff in subcrop_shape_diffs],
                        shape=patch_coords.shape,
                    ),
                    padding_val=self.padding_val,
                )
                output.append(padded_subcrop)
            else:
                # otherwise, get the patch data directly as a view (saves a copy later)
                output.append(
                    flex_crop(
                        image=image,
                        patch=patch_coords,
                        padding_val=self.padding_val,
                        force_copy=False,
                    )
                )

        # if we don't need to (or cannot) rebuild the patch grid, return the array
        if mask is not None or not self.make_contiguous:
            if self.make_contiguous:
                if isinstance(image, np.ndarray):
                    output = np.stack(output)
                else:
                    output = torch.stack(output)
            return output

        # otherwise, copy the patch data into the newly allocated grid
        offset_steps = self._get_offset_steps_from_abs_patch_shape(patch_shape)
        patch_idx_arrays = list(zip(*patch_coords_map.keys()))
        init_coords = next(iter(patch_coords_map.keys())) if patch_coords_map else tuple()
        assert all([
            all([
                (idx - init_coords[d]) % offset_steps[d] == 0
                for idx in patch_idx_arrays[d]
            ]) for d in range(len(patch_shape))
        ])
        grid_shape = tuple([
            len(np.unique(patch_idx_arrays[d])) for d in range(len(patch_shape))
        ])
        output_shape = grid_shape + image.shape[:-len(patch_shape)] + patch_shape
        if isinstance(image, torch.Tensor):
            output_grid = torch.empty(output_shape, dtype=image.dtype, device=image.device)
        else:
            output_grid = np.empty(output_shape, dtype=image.dtype)
        for idxs, patch in zip(patch_coords_map.keys(), output):
            grid_loc = tuple([  # note: this is where we deduce the patch coords inside the grid
                (idxs[d] - init_coords[d]) // offset_steps[d]
                for d in range(len(patch_shape))
            ])
            output_grid[grid_loc] = patch
        return output_grid

    def _check_if_mask_crop_is_good(
        self,
        mask: typing.Union[np.ndarray, torch.Tensor],
        patch_coord: patches.PatchCoord,
    ) -> bool:
        """Returns whether the mask crop at the specified location passes the IoU threshold or not."""
        mask_crop = flex_crop(mask, patch_coord, force_copy=False)
        if isinstance(mask, np.ndarray):
            mask_area = np.count_nonzero(mask_crop)
        elif isinstance(mask, torch.Tensor):
            mask_area = torch.count_nonzero(mask_crop).item()
        else:
            raise TypeError
        return mask_area >= patch_coord.size * self.min_mask_iou

    def _find_init_patch_coord(
        self,
        patch_shape: typing.Tuple[int, ...],
        offset_steps: typing.Tuple[int, ...],
        mask: typing.Optional[typing.Union[np.ndarray, torch.Tensor]] = None,
    ) -> typing.Optional[typing.Tuple[int, ...]]:
        """Returns the initial coordinates from where to start extracting patches."""
        assert len(patch_shape) == len(offset_steps)
        init_idxs = tuple([
            -(patch_shape[d] - offset_steps[d]) if self.offset_overlap else 0
            for d in range(len(patch_shape))
        ])
        if mask is None:
            return init_idxs
        assert isinstance(mask, (np.ndarray, torch.Tensor)), \
            "mask type should be np.ndarray or torch.Tensor"
        assert mask.ndim == len(patch_shape), "mask ndim should be equal to spatial ndim only!"
        assert mask.ndim == len(patch_shape)
        spatial_dim_ranges = tuple([
            range(init_idxs[d], mask.shape[d]) for d in range(len(patch_shape))
        ])
        found_idxs = None  # will remain none if mask never provides a good first hit
        for idxs in itertools.product(*spatial_dim_ranges):
            curr_patch = patches.PatchCoord(idxs, shape=patch_shape)
            if self._check_if_mask_crop_is_good(mask, curr_patch):
                found_idxs = tuple(idxs)
                break
        return found_idxs

    def get_patch_grid_ranges(
        self,
        image_shape: typing.Tuple[int, ...],
    ) -> typing.Tuple[range]:
        """Returns the grid ranges that can be used to define the patch coords for an image shape.

        image_shape: the (spatial) shape of the image to cut into patches. The dim count of
            this tuple must be equal to the spatial dim count of the patches.

        Returns:
            An array of dim-wise ranges that can be used to determine the (top-left) location of
            patches that would be extracted from an image of the specified shape when no mask
            is considered.
        """
        assert isinstance(image_shape, (list, tuple)), "unexpected image shape type"
        patch_shape = self._get_abs_patch_shape_from_image_shape(image_shape)
        offset_steps = self._get_offset_steps_from_abs_patch_shape(patch_shape)
        spatial_start_idxs = self._find_init_patch_coord(patch_shape, offset_steps)
        spatial_end_idxs = tuple([
            (image_shape[d] - offset_steps[d] + 1)
            if self.offset_overlap else (image_shape[d] - patch_shape[d] + 1)
            for d in range(len(patch_shape))
        ])
        return tuple([
            range(spatial_start_idxs[d], spatial_end_idxs[d], offset_steps[d])
            for d in range(len(patch_shape))
        ])

    def get_patch_coords(
        self,
        image: typing.Union[np.ndarray, torch.Tensor],
        mask: typing.Optional[typing.Union[np.ndarray, torch.Tensor]] = None,
    ) -> typing.Dict[typing.Tuple[int, ...], patches.PatchCoord]:
        """Returns the coordinates of the patches that would be extracted from the given input.

        When calling the transform with a mask argument or with `make_contiguous` as false, the
        order of the extracted patches provided as output will be the same as the order of the
        patches provided here.

        Args:
            image: the image to cut into patches. This image can have more dimensions than the
                number of dimensions provided in the patch size given in the constructor.
            mask: the mask (ROI) indicating the area where to extract the patches (can be ``None``).
                If the mask is given, its number of dimensions must be equal to the number of patch
                dimensions, which should also be equal to the number of spatial dimensions we will
                be cutting inside the image.

        Returns:
            A map indexing patch locations (top-left coordinates) to patch coordinates objects.
        """
        assert isinstance(image, (np.ndarray, torch.Tensor)), \
            "image type should be np.ndarray or torch.Tensor"
        patch_shape = self._get_abs_patch_shape_from_image_shape(image.shape)
        offset_steps = self._get_offset_steps_from_abs_patch_shape(patch_shape)
        spatial_start_idxs = self._find_init_patch_coord(patch_shape, offset_steps, mask)
        im_shape = tuple(image.shape[-len(patch_shape):])
        cache_key = (patch_shape, offset_steps, spatial_start_idxs, im_shape)
        if mask is None and cache_key in self._patch_coords_cache:
            return self._patch_coords_cache[cache_key]  # shortcut for when we have no mask to check!
        spatial_end_idxs = tuple([
            (im_shape[d] - offset_steps[d] + 1)
            if self.offset_overlap else (im_shape[d] - patch_shape[d] + 1)
            for d in range(len(patch_shape))
        ])
        spatial_dim_ranges = tuple([
            range(spatial_start_idxs[d], spatial_end_idxs[d], offset_steps[d])
            for d in range(len(patch_shape))
        ])
        output = {}
        for idxs in itertools.product(*spatial_dim_ranges):
            idxs = tuple(idxs)
            assert idxs not in output  # we should never have duplicates...?
            curr_patch = patches.PatchCoord(idxs, shape=patch_shape)
            if mask is None or self._check_if_mask_crop_is_good(mask, curr_patch):
                output[idxs] = curr_patch
        if mask is None:
            self._patch_coords_cache[cache_key] = output
        return output

    def __repr__(self) -> str:
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(patch_shape={self.patch_shape}, patch_overlap={self.patch_overlap}, " \
            f"min_mask_iou={self.min_mask_iou}, offset_overlap={self.offset_overlap}, " \
            f"padding_val={self.padding_val}, make_contiguous={self.make_contiguous})"
