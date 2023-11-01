import numpy as np

import hardpicks.data.patchify as patchify
import hardpicks.utils.patch_utils as patches


def test_flex_crop_regular_tensor():
    image = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))  # B x C x H x W
    patch = patches.PatchCoord((2, 3), bottom_right=(5, 6))
    crop = patchify.flex_crop(
        image=image,
        patch=patch,
    )
    assert crop.shape == (3, 4, 3, 3)  # B x C x patchsize
    assert np.array_equal(crop, image[:, :, 2:5, 3:6])
    assert crop.base is not None  # i.e. it is a view
    crop_copy = patchify.flex_crop(
        image=image,
        patch=patch,
        force_copy=True,
    )
    assert crop_copy.base is None  # in this case, it should now be a copy
    assert np.array_equal(crop_copy, crop)
    crop[:, :, -1, -1] = -1
    assert not np.array_equal(crop_copy, crop)
    assert (image[:, :, 4, 5] == -1).all()
    # now, let's test the oob behavior
    patch = patches.PatchCoord((2, 3), bottom_right=(7, 5))
    crop = patchify.flex_crop(
        image=image,
        patch=patch,
    )
    assert crop.shape == (3, 4, 5, 2)
    assert np.array_equal(crop[:, :, :3, :], image[:, :, 2:, 3:5])
    assert crop.base is None  # we added padding, it cannot be a view


def test_flex_crop_opencv_image():
    image = np.arange(600, dtype=np.int16).reshape((10, 20, 3))
    patch = patches.PatchCoord((2, 3), bottom_right=(8, 6))
    crop = patchify.flex_crop(
        image=image,
        patch=patch,
    )
    assert crop.shape == (6, 3, 3)  # patchsize x C
    assert np.array_equal(crop, image[2:8, 3:6, :])
    assert crop.base is not None  # i.e. it is a view


def test_patchify_no_overlap_no_grid():
    img = np.arange(100).reshape(10, 10)
    patch_shape = (2, 2)
    patcher = patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=False,
    )
    out = patcher(img)
    assert isinstance(out, list) and len(out) == 25
    assert np.array_equal(out[0], img[0:2, 0:2])
    assert np.array_equal(out[1], img[0:2, 2:4])
    assert np.array_equal(out[5], img[2:4, 0:2])
    assert np.array_equal(out[-1], img[8:10, 8:10])

    img = np.arange(100).reshape(10, 10)
    patch_shape = (2, 4)
    patcher = patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=False,
    )
    out = patcher(img)
    assert isinstance(out, list) and len(out) == 10
    assert np.array_equal(out[0], img[0:2, 0:4])
    assert np.array_equal(out[1], img[0:2, 4:8])
    assert np.array_equal(out[2], img[2:4, 0:4])
    assert np.array_equal(out[-1], img[8:10, 4:8])


def test_patchify_no_overlap_with_grid():
    img = np.arange(100).reshape(10, 10)
    patch_shape = (2, 2)
    patcher = patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=True,
    )
    out = patcher(img)
    assert isinstance(out, np.ndarray) and out.shape == (5, 5, 2, 2)
    assert np.array_equal(out[0, 0], img[0:2, 0:2])
    assert np.array_equal(out[0, 1], img[0:2, 2:4])
    assert np.array_equal(out[0, 2], img[0:2, 4:6])
    assert np.array_equal(out[1, 0], img[2:4, 0:2])
    assert np.array_equal(out[1, 1], img[2:4, 2:4])
    assert np.array_equal(out[-1, -1], img[8:10, 8:10])

    img = np.arange(100).reshape(10, 10)
    patch_shape = (2, 4)
    patcher = patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=True,
    )
    out = patcher(img)
    assert isinstance(out, np.ndarray) and out.shape == (5, 2, 2, 4)
    assert np.array_equal(out[0, 0], img[0:2, 0:4])
    assert np.array_equal(out[0, 1], img[0:2, 4:8])
    assert np.array_equal(out[1, 0], img[2:4, 0:4])
    assert np.array_equal(out[1, 1], img[2:4, 4:8])
    assert np.array_equal(out[-1, -1], img[8:10, 4:8])


def test_patchify_overlap_with_grid():
    img = np.arange(1200).reshape(30, 40)
    patch_shape = (4, 8)
    patcher = patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0.25,
        offset_overlap=False,
        make_contiguous=True,
    )
    out = patcher(img)
    assert isinstance(out, np.ndarray) and out.shape == (9, 6, 4, 8)
    assert np.array_equal(out[0, 0], img[0:4, 0:8])
    assert np.array_equal(out[0, 1], img[0:4, 6:14])
    assert np.array_equal(out[1, 0], img[3:7, 0:8])
    assert np.array_equal(out[1, 1], img[3:7, 6:14])
    assert np.array_equal(out[-1, -1], img[24:28, 30:38])


def test_patchify_overlap_with_grid_and_offset():
    img = np.arange(1200).reshape(30, 40)
    patch_shape = (4, 8)
    patcher = patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0.25,
        offset_overlap=True,
        padding_val=-1,
        make_contiguous=True,
    )
    out = patcher(img)
    assert isinstance(out, np.ndarray) and out.shape == (10, 7, 4, 8)
    assert (out[0, 0][:, :2] == -1).all()
    assert (out[0, 0][:1, :] == -1).all()
    assert np.array_equal(out[0, 0, 1:, 2:], img[0:3, 0:6])
    assert (out[0, 1][:1, :] == -1).all()
    assert np.array_equal(out[0, 1][1:, :], img[0:3, 4:12])
    assert (out[1, 0][:, :2] == -1).all()
    assert np.array_equal(out[1, 0][:, 2:], img[2:6, 0:6])
    assert (out[-1, -1][:, 6:] == -1).all()
    assert np.array_equal(out[-1, -1][:, :6], img[26:30, 34:40])


def test_patchify_with_mask():
    img = np.arange(1200).reshape(30, 40)
    patch_shape = (4, 8)
    patcher = patchify.Patchify(
        patch_shape=patch_shape,
        patch_overlap=0,
        offset_overlap=False,
        make_contiguous=False,
    )
    mask = np.zeros((30, 40))
    mask[16:, 10:] = 1
    out = patcher(img, mask)
    assert isinstance(out, list) and len(out) == 9
    assert np.array_equal(out[0], img[16:20, 10:18])
    assert np.array_equal(out[1], img[16:20, 18:26])
    assert np.array_equal(out[2], img[16:20, 26:34])
    assert np.array_equal(out[3], img[20:24, 10:18])
    assert np.array_equal(out[-1], img[24:28, 26:34])
