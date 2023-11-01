import numpy as np
import pytest
import torchvision.transforms.functional

import hardpicks.data.transforms as generic_transforms
import hardpicks.utils.patch_utils as patch_utils


def test_random_stochastic_op_wrapper():
    class Counter:
        call_count = 0

        def count(self):
            self.call_count += 1

    counter = Counter()
    count_prob = 0.1
    wrapped_counter = generic_transforms.stochastic_op_wrapper(
        operation=counter.count,
        prob=count_prob,
    )
    real_call_count = 100000
    for _ in range(real_call_count):
        wrapped_counter()
    assert counter.call_count > 0
    # really, we just wanna make sure the probability isn't inverted... (x vs 1-x)
    assert np.isclose(counter.call_count / real_call_count, count_prob, atol=0.015)


def test_get_one_hot_encoding():
    # start with a 1D test; 7 labels over 10 classes (some of which are never seen)
    label_vec = np.asarray([0, 1, 2, 3, 2, 1, 2])
    out = generic_transforms.get_one_hot_encoding(label_vec, class_count=10)
    assert out.shape == (10, 7)
    for input_idx, label in enumerate(label_vec):
        assert out[:, input_idx].sum() == 1
        assert out[label, input_idx] == 1
    # now, try a 2d array and make sure the results are still good
    label_map = np.random.randint(0, 5, size=(15, 19))
    out = generic_transforms.get_one_hot_encoding(label_map, class_count=5)
    assert out.shape == (5, 15, 19)
    assert (out.sum(axis=0) == 1).all()
    for row_idx in range(label_map.shape[0]):
        for col_idx in range(label_map.shape[1]):
            assert out[label_map[row_idx, col_idx], row_idx, col_idx] == 1
    # finally, try with an array that has out-of-bound indices
    label_vec = np.random.randint(-3, 10, size=(100,))
    out = generic_transforms.get_one_hot_encoding(label_vec, class_count=7)
    assert out.shape == (7, 100)
    for input_idx, label in enumerate(label_vec):
        if label < 0 or label >= 7:
            assert out[:, input_idx].sum() == 0
        else:
            assert out[:, input_idx].sum() in [0, 1]
            assert out[label, input_idx] == 1


@pytest.fixture
def image():
    return np.arange(20000, dtype=np.int32).reshape((100, 200))  # H,W = 100,200


def test_random_resized_crop__abs_without_resize(image):
    im_h, im_w = image.shape
    cropper_square = generic_transforms.RandomResizedCrop(
        output_size=None,
        input_size=(50, 100),  # min_edge_size, max_edge_size
        ratio=1,  # crops regions must be perfectly square
        padding_val=-1,
    )
    cropper_widerect = generic_transforms.RandomResizedCrop(
        output_size=None,
        input_size=(25, 100),  # min_edge_size, max_edge_size
        ratio=2,  # crops regions must be approx. twice as wide as tall
        padding_val=-1,
    )
    cropper_anyrect = generic_transforms.RandomResizedCrop(
        output_size=None,
        input_size=(25, 100),  # min_edge_size, max_edge_size
        ratio=(0.5, 2.0),
        padding_val=-1,
    )
    for _ in range(100):  # run 100 random crops on the image
        crop = cropper_square(image)
        assert crop.ndim == 2  # since image is 2D
        assert crop.shape[0] == crop.shape[1]  # since ratio = 1
        assert 50 <= max(crop.shape) <= 100  # since square crops
        assert not (crop == -1).any()  # since crop size fits in image
        # now, let's just verify that the data is truly intact
        crop_tl_idx = crop[0, 0].item()
        crop_tl_row, crop_tl_col = crop_tl_idx // im_w, crop_tl_idx % im_w
        orig_crop_coords = patch_utils.PatchCoord((crop_tl_row, crop_tl_col), shape=crop.shape)
        assert np.array_equal(crop, image[orig_crop_coords.slice])
        crop = cropper_widerect(image)
        assert crop.ndim == 2  # since image is 2D
        assert 25 <= max(crop.shape) and min(crop.shape) <= 100  # since rect
        out_ratio = crop.shape[1] / crop.shape[0]
        # note: the aspect ratio range is usually not perfectly matched (+/- 1px)
        min_allowed_ratio = ((min(crop.shape) - 1) * 2) / (min(crop.shape) + 1)
        max_allowed_ratio = ((max(crop.shape) + 1) * 2) / (max(crop.shape) - 1)
        assert min_allowed_ratio <= out_ratio <= max_allowed_ratio
        crop = cropper_anyrect(image)
        assert crop.ndim == 2  # since image is 2D
        assert 25 <= max(crop.shape) and min(crop.shape) <= 100  # since rect
        out_ratio = crop.shape[1] / crop.shape[0]
        min_allowed_ratio = ((min(crop.shape) - 1) * 0.5) / (min(crop.shape) + 1)
        max_allowed_ratio = ((max(crop.shape) + 1) * 2) / (max(crop.shape) - 1)
        assert min_allowed_ratio <= out_ratio <= max_allowed_ratio


def test_random_resized_crop__rel_with_resize(image):
    im_h, im_w = image.shape
    cropper = generic_transforms.RandomResizedCrop(
        output_size=(im_h * 2, im_w * 2),
        input_size=((0.25, 0.5), (0.75, 0.5)),  # (min_h, min_w), (max_h, max_w)
        ratio=None,  # will randomly sample in the tuple range given above
        padding_val=-1,
        interp=torchvision.transforms.functional.InterpolationMode.NEAREST,
    )
    for _ in range(100):  # run 100 random crops on the image
        crop = cropper(image)
        assert crop.ndim == 2  # since image is 2D
        assert crop.shape == (im_h * 2, im_w * 2)  # we asked for this
        assert not (crop == -1).any()  # since crop size fits in image
        # now, let's just verify that the data is truly intact
        crop_tl_idx = crop[0, 0].item()
        crop_tl_row, crop_tl_col = crop_tl_idx // im_w, crop_tl_idx % im_w
        crop_br_idx = crop[-1, -1].item()
        crop_br_row, crop_br_col = (crop_br_idx // im_w) + 1, (crop_br_idx % im_w) + 1
        orig_crop_coords = patch_utils.PatchCoord(
            (crop_tl_row, crop_tl_col), bottom_right=(crop_br_row, crop_br_col),
        )
        assert np.array_equal(np.unique(crop), np.unique(image[orig_crop_coords.slice]))
        assert 0.25 * im_h <= orig_crop_coords.shape[0] <= 0.75 * im_h
        assert orig_crop_coords.shape[1] == 0.5 * im_w
