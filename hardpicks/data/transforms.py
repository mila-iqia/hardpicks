"""Generic data transformation utilities."""
import functools
import math
import typing

import einops
import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional as torchvis_transfunc

import hardpicks.data.patchify as patchify
import hardpicks.utils.patch_utils as patch_utils
import hardpicks.utils.hp_utils as hp_utils


def _stochastic_op(
    _operation: typing.Callable,
    _prob: float,
    **kwargs,
):
    """Full definition of the stochastic op 'wrapper' that can be pickled."""
    if np.isclose(_prob, 1.0) or np.random.rand() < _prob:
        return _operation(**kwargs)


def stochastic_op_wrapper(
    operation: typing.Callable,
    prob: float,
):
    """Wraps a given callable operation to make it called/skipped stochastically."""
    assert 0 < prob <= 1.0, f"invalid operation probability ({prob})"
    return functools.partial(
        _stochastic_op,
        _operation=operation,
        _prob=prob,
    )


def get_one_hot_encoding(
    label_map: typing.Union[np.ndarray, torch.Tensor],
    class_count: int,
) -> typing.Union[np.ndarray, torch.Tensor]:
    """Returns a one-hot encoded version of the provided n-dim array of label indices.

    For a given N x M x ... label array, the returned one-hot encoded label map will have the label
    dimension in its 0th dimension, i.e. we will return a C x N x M x ... array.

    If out-of-bounds label indices are present in the given label map, they will be assigned the
    full-zero encoding instead of a one-hot encoding.
    """
    assert class_count > 0, "unexpected class count for one-hot array shape allocation"
    # we'll create an array of C+1 rows that are one-hot vectors for each of the C classes and an
    # extra vector filled with zeros that will be used for all label indices that fall out of bounds
    row_count = class_count + 1
    if isinstance(label_map, np.ndarray):
        one_hot_matrix = np.eye(row_count, class_count, dtype=np.int16)
        # reallocate the label map with all bad pixels assigned to the zero encoding row index
        label_map = np.where(
            np.logical_or(label_map < 0, label_map >= class_count),
            np.full_like(label_map, fill_value=class_count),
            label_map,
        )
    else:
        assert isinstance(label_map, torch.Tensor), "unexpected array type"
        one_hot_matrix = torch.eye(row_count, class_count, dtype=torch.int16, device=label_map.device)
        # reallocate the label map with all bad pixels assigned to the zero encoding row index
        label_map = torch.where(
            torch.logical_or(label_map < 0, label_map >= class_count),
            torch.full_like(label_map, fill_value=class_count),
            label_map,
        ).long()
    flat_labels = label_map.reshape(-1)  # keeps everything in order but makes lookups possible
    flat_onehots = one_hot_matrix[flat_labels]  # we get a N x C array
    return einops.rearrange(  # return it to C x (...)
        flat_onehots, "n c -> c n",
    ).reshape((class_count, *label_map.shape))


class RandomResizedCrop:
    """Returns a resized crop of a randomly selected image region.

    Compared to the torchvision version of the same operation, this version allows the user to
    specify a range of patch sizes used for input instead of only using a scale value relative to
    the original image size. This allows for better control over the overall scaling applied to
    crops taken from images of varying sizes.

    If the image is provided as a numpy array with two or three dimensions, it will be cropped under
    the assumption that it is an OpenCV-like image with a number of channels of 1 to 4 and that the
    channel dimension is the last in the array shape (e.g. H x W x C), resulting in patches with
    a similar dimension ordering. Otherwise, the image will be cropped under the assumption that we
    are doing a 2D spatial crop and that the spatial dimensions are the last dimensions in the
    array shape (e.g. for a 4-dim tensor B x C x H x W and a 2D crop size, the output will be
    B x C x [patch.shape]).

    Attributes:
        output_size: size of the output crop, provided as a single element (``edge_size``) or as a
            two-element tuple or list (``[height, width]``). If integer values are used, the size is
            assumed to be absolute. If floating point values are used (i.e. in [0,1]), the output
            size is assumed to be relative to the original image size, and will be determined at
            execution time for each sample. If set to ``None``, the crop will not be resized.
        input_size: range of the input region sizes, provided as a pair of elements
            (``[min_edge_size, max_edge_size]``) or as a pair of tuples or lists
            (``[[min_height, min_width], [max_height, max_width]]``). If the pair-of-pairs format is
            used, the ``ratio`` argument cannot be used. If integer values are used, the ranges are
            assumed to be absolute. If floating point values are used (i.e. in [0,1]), the ranges
            are assumed to be relative to the original image size, and will be determined at
            execution time for each sample.
        ratio: range of minimum/maximum input region aspect ratios to use. This argument cannot be
            used if the pair-of-pairs format is used for the ``input_size`` argument.
        probability: the probability that the transformation will be applied when called; if not
            applied, the returned image will be the original.
        padding_val: value used for padding when we get a crop that is outside the image bounds.
        interp: interpolation flag forwarded to ``torchvision.transforms.functional.resize``.
    """

    def __init__(
        self,
        output_size: typing.Optional[typing.Union[typing.Union[int, float], typing.Tuple]],
        input_size: typing.Union[typing.Tuple, typing.Tuple[typing.Tuple, typing.Tuple]] = (0.08, 1.0),
        ratio: typing.Optional[typing.Union[typing.Tuple[float, float], float]] = (0.75, 1.33),
        probability: float = 1.0,
        padding_val: typing.Union[int, float] = 0,
        interp: torchvis_transfunc.InterpolationMode = torchvis_transfunc.InterpolationMode.BILINEAR,
    ):
        """Validates and initializes crop parameters.

        Args:
            output_size: size of the output crop, provided as a single element (``edge_size``) or as a
                two-element tuple or list (``[height, width]``). If integer values are used, the size is
                assumed to be absolute. If floating point values are used (i.e. in [0,1]), the output
                size is assumed to be relative to the original image size, and will be determined at
                execution time for each sample. If set to ``None``, the crop will not be resized.
            input_size: range of the input region sizes, provided as a pair of elements
                (``[min_edge_size, max_edge_size]``) or as a pair of tuples or lists
                (``[[min_height, min_width], [max_height, max_width]]``). If the pair-of-pairs format is
                used, the ``ratio`` argument cannot be used. If integer values are used, the ranges are
                assumed to be absolute. If floating point values are used (i.e. in [0,1]), the ranges
                are assumed to be relative to the original image size, and will be determined at
                execution time for each sample.
            ratio: range of minimum/maximum input region aspect ratios to use. This argument cannot be
                used if the pair-of-pairs format is used for the ``input_size`` argument.
            probability: the probability that the transformation will be applied when called; if not
                applied, the returned image will be the original.
            padding_val: value used for padding when we get a crop that is outside the image bounds.
            interp: interpolation flag forwarded to ``torchvision.transforms.functional.resize``.
        """
        if isinstance(output_size, str):
            output_size = hp_utils.get_array_from_input_that_could_be_a_string(output_size)
        if output_size is None:
            # if the output size is not specified at all, we will just skip the resize operation
            self.output_size = None
        elif isinstance(output_size, (tuple, list)):
            # ...otherwise, it might be specified as a 2-elem tuple of (height, width)
            assert len(output_size) == 2, \
                "expected output size to be two-element list or tuple, or single scalar"
            assert (
                all([isinstance(s, int) and s > 0 for s in output_size])
                or all([isinstance(s, float) and s > 0 for s in output_size])
            ), "expected size pair elements to be the same type and greater than zero"
            self.output_size = tuple(output_size)
        else:  # if only a scalar is given, assume it's the edge size
            assert (
                (isinstance(output_size, int) and output_size > 0)
                or (isinstance(output_size, float) and output_size > 0)
            ), "expected size scalar to be int/float and greater than zero"
            self.output_size = (output_size, output_size)

        if isinstance(input_size, str):
            input_size = hp_utils.get_array_from_input_that_could_be_a_string(input_size)
        if isinstance(ratio, str):
            ratio = hp_utils.get_array_from_input_that_could_be_a_string(ratio)
        assert isinstance(input_size, (tuple, list)) and len(input_size) == 2, \
            "expected input size to be provided as a pair of elements"
        if all([isinstance(s, int) for s in input_size]) or all([isinstance(s, float) for s in input_size]):
            # if we are here, then the input size is a tuple of sizes (not a tuple-of-tuples)
            # ... this means we will NEED the ratio argument to be available
            assert ratio is not None, "when specifying input as tuple of sizes, we need ratio!"
            if isinstance(ratio, (tuple, list)):
                assert len(ratio) == 2, "invalid ratio tuple/list length (expected two elements)"
                assert all([isinstance(r, (int, float)) and r > 0 for r in ratio]), \
                    "expected ratio pair elements to be greater-than-zero scalar values"
                self.ratio = (float(min(ratio)), float(max(ratio)))
            else:
                assert isinstance(ratio, (int, float)) and ratio > 0, \
                    "invalid ratio type (should be tuple of scalars or single scalar)"
                self.ratio = (float(ratio), float(ratio))
            # next, check that the input size value ranges are valid
            for s in input_size:
                assert ((isinstance(s, float) and 0 < s <= 1) or (isinstance(s, int) and s > 0)), \
                    f"invalid input size value ({str(s)})"
            min_edge_size, max_edge_size = input_size
            assert min_edge_size <= max_edge_size, "inverted min_edge_size and max_edge_size"
            self.input_size = ((min_edge_size, min_edge_size), (max_edge_size, max_edge_size))
        else:
            # otherwise, the input size must be a tuple of tuples, and we CANNOT have a ratio tuple
            assert all([isinstance(s, (tuple, list)) and len(s) == 2 for s in input_size]), \
                "expected input size to be pair of min/max (height, width) values"
            assert ratio is None, \
                "cannot specify input sizes in two-element tuples/lists and also provide aspect ratios"
            # next, check that the input size types are same-type scalars and their values are valid
            for t in input_size:
                for s in t:
                    assert isinstance(s, (int, float)) and isinstance(s, type(input_size[0][0])), \
                        "input sizes should all be same type, either int or float"
                    assert ((isinstance(s, float) and 0 < s <= 1) or (isinstance(s, int) and s > 0)), \
                        f"invalid input size value ({str(s)})"
            (min_height, min_width), (max_height, max_width) = input_size
            assert min_height <= max_height, "inverted min_height and max_height"
            assert min_width <= max_width, "inverted min_width and max_width"
            self.input_size = ((min_height, min_width), (max_height, max_width))
            self.ratio = None  # ignored since input_size contains all necessary info

        assert 0 <= probability <= 1, "invalid probability value (should be in [0,1])"
        self.probability = probability
        self.padding_val = padding_val
        self.interp = interp

    def get_patch(
        self,
        image_shape: typing.Tuple[int, int],
    ) -> patch_utils.PatchCoord:
        """Returns the patch coordinates of a possible crop for the given image shape.

        Args:
            image_shape: the image shape inside which to generate the crop.

        Returns:
            The patch coords object of a randomly selected crop region in the image region. Note that
            the size of this patch corresponds to the crop BEFORE resizing!
        """
        assert len(image_shape) == 2
        image_height, image_width = image_shape
        if self.probability < 1 and np.random.uniform(0, 1) > self.probability:
            return patch_utils.PatchCoord((0, 0), shape=image_shape)

        # if we are using fixed min/max crop shapes, we can sample those right away
        if self.ratio is None:
            target_height = np.random.uniform(self.input_size[0][0], self.input_size[1][0])
            target_width = np.random.uniform(self.input_size[0][1], self.input_size[1][1])
            if isinstance(self.input_size[0][0], float):  # if using rel sizes, get abs values now
                target_width *= image_width
                target_height *= image_height
            target_width = int(round(target_width))
            target_height = int(round(target_height))

        # otherwise, if we are using the aspect ratio tuple, go with the original area-based computation
        elif isinstance(self.input_size[0][0], (int, float)):
            if isinstance(self.input_size[0][0], float):
                area = image_height * image_width
                target_area = np.random.uniform(self.input_size[0][0], self.input_size[1][0]) * area
            else:
                target_area = np.random.uniform(self.input_size[0][0], self.input_size[1][0]) ** 2
            aspect_ratio = np.random.uniform(*self.ratio)
            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

        else:
            raise RuntimeError("unhandled crop strategy")

        # if the crop is smaller than the input image, we can place it randomly in-bounds
        if target_width <= image_width and target_height <= image_height:
            target_row = np.random.randint(
                min(0, image_height - target_height),
                max(0, image_height - target_height) + 1,
            )
            target_col = np.random.randint(
                min(0, image_width - target_width),
                max(0, image_width - target_width) + 1,
            )
        else:  # otherwise, use a centered crop (in case we hit any edge)
            target_row = (image_height - target_height) // 2
            target_col = (image_width - target_width) // 2

        return patch_utils.PatchCoord((target_row, target_col), shape=(target_height, target_width))

    def crop_and_resize(
        self,
        image: typing.Union[np.ndarray, torch.Tensor],
        crop_patch: patch_utils.PatchCoord,
    ) -> torch.Tensor:
        """Runs the actual crop-and-resize combo operation for a given patch in the given image.

        Args:
            image: the image to generate the crop from, in either H x W x C (OpenCV-like) or
                C x H x W (torch-like) format.

        Returns:
            The resized crop of the specified region as a `torch.Tensor` in C x H x W format.
        """
        crop = patchify.flex_crop(
            image,
            patch=crop_patch,
            padding_val=self.padding_val,
            force_copy=self.output_size is None,  # copy now if not resizing afterwards
        )
        if isinstance(crop, np.ndarray):
            crop = torch.from_numpy(crop)
        # if we don't have to resize anything, return the copy directly
        if self.output_size is None:
            return crop

        # otherwise, use the cropped view to return a resized version of the crop
        output_height, output_width = self.output_size
        image_height, image_width = image.shape[-2], image.shape[-1]
        if isinstance(output_height, float):  # if using relative coords, get the abs ones again
            output_height = max(int(round(output_height * image_height)), 1)
            output_width = max(int(round(output_width * image_width)), 1)

        assert crop.ndim in [2, 3, 4], "unsupported tensor dim for torch resize op"
        need_channel_dim, need_batch_dim = crop.ndim < 3, crop.ndim < 4
        if need_channel_dim:
            crop = crop.unsqueeze(0)  # this will add the channel dimension if missing
        if need_batch_dim:
            crop = crop.unsqueeze(0)  # this will add the batch dimension if missing
        out = torchvis_transfunc.resize(
            img=crop,
            size=[output_height, output_width],
            interpolation=self.interp,
        )
        assert isinstance(out, torch.Tensor)
        if need_batch_dim:
            out = out.squeeze(0)
        if need_channel_dim:
            out = out.squeeze(0)
        return out

    def __call__(
        self,
        image: typing.Union[PIL.Image.Image, np.ndarray, torch.Tensor],
    ) -> typing.Union[np.ndarray, torch.Tensor]:
        """Extracts and returns a random (resized) crop from the provided image.

        Args:
            image: the image to generate the crop from, in either H x W x C (OpenCV-like) or
                C x H x W (torch-like) format.

        Returns:
            The randomly selected and resized crop as a `torch.Tensor` in C x H x W format.
        """
        assert isinstance(image, (PIL.Image.Image, np.ndarray, torch.Tensor)), \
            "image type should be torch tensor, numpy array, or PIL image"
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image)
        assert image.ndim >= 2, "cannot do 2d crop in a non-2D array?"
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] in [1, 2, 3, 4]:
            # special handling for OpenCV-like arrays (where the channel is the last dimension)
            image = np.transpose(image, (2, 0, 1))  # convert to C x H x W format
        crop_patch = self.get_patch((image.shape[-2], image.shape[-1]))
        return self.crop_and_resize(image, crop_patch)

    def __repr__(self) -> str:
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(output_size={self.output_size}, input_size={self.input_size}, ratio={self.ratio}, " + \
            f"probability={self.probability}, padding_val={self.padding_val}, interp={self.interp})"
