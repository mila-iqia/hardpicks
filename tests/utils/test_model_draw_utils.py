import einops
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hardpicks.models.constants import DONTCARE_SEGM_MASK_LABEL
from hardpicks.utils.model_draw_utils import (
    get_most_likely_class_and_lightened_inputs_image_array,
    get_most_likely_class_image_array,
    VALIDATION_COORDINATES_COLOR,
    overlay_image_label_map_with_with_colored_points,
)


@pytest.fixture
def class_colors(number_of_classes):
    all_colors = np.array(
        [
            [0, 100, 1],
            [240, 231, 140],
            [255, 255, 0],
            [138, 43, 225],
            [187, 85, 212],
            [238, 130, 239],
            [254, 0, 0],
        ],
        dtype=np.uint8,
    )
    return all_colors[:number_of_classes]


@pytest.fixture
def height():
    return 210


@pytest.fixture
def width():
    return 210


@pytest.fixture
def number_of_classes():
    return 2


@pytest.fixture
def class_probability_map(number_of_classes, height, width):
    torch.manual_seed(23423)
    raw_scores = torch.rand(number_of_classes, height, width)
    probabilities = F.softmax(raw_scores, dim=0).cpu().numpy()
    return probabilities


@pytest.fixture
def input_labels_map(number_of_classes, height, width):
    np.random.seed(2423)
    classes = list(range(number_of_classes)) + [DONTCARE_SEGM_MASK_LABEL]
    labels_map = np.random.choice(classes, [height, width])
    return labels_map


@pytest.mark.parametrize("prediction_threshold", [None, 0.9])
def test_smoke_get_most_likely_class_and_lightened_inputs_image_array(
    class_probability_map, input_labels_map, class_colors, prediction_threshold
):
    get_most_likely_class_and_lightened_inputs_image_array(
        class_probability_map, input_labels_map, class_colors, prediction_threshold
    )


@pytest.fixture
def coordinates(height, width):
    np.random.seed(34523)
    number_of_points = 40
    list_i = np.random.randint(0, height, number_of_points)
    list_j = np.random.randint(0, width, number_of_points)
    coordinates = einops.rearrange([list_i, list_j], "d n -> n d")

    return coordinates


@pytest.mark.parametrize("prediction_threshold", [None, 0.9])
def test_overlay_image_label_map_with_with_colored_points(
    class_probability_map, class_colors, prediction_threshold, coordinates
):
    image_label_map = get_most_likely_class_image_array(
        class_probability_map, class_colors, prediction_threshold
    )
    image_label_map = overlay_image_label_map_with_with_colored_points(
        image_label_map, coordinates, VALIDATION_COORDINATES_COLOR
    )

    set_of_coordinates = set([tuple(c) for c in coordinates])
    for i in range(image_label_map.shape[0]):
        for j in range(image_label_map.shape[1]):
            color_error = np.linalg.norm(
                image_label_map[i, j] - VALIDATION_COORDINATES_COLOR
            )

            if (i, j) in set_of_coordinates:
                assert color_error < 0.1
            else:
                assert color_error >= 1.0
