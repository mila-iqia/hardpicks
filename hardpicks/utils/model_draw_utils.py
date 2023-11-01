"""This module implements utility functions to draw useful artifacts while training a model."""
from typing import List

import numpy as np
import torch
from einops import reduce, rearrange
from torch.nn import functional as F

from hardpicks.metrics.classification_utils import get_class_index_from_probabilities, \
    CANNOT_PREDICT_INDICATOR
from hardpicks.models.constants import DONTCARE_SEGM_MASK_LABEL
from hardpicks.utils import draw_utils

UNDEFINED_CLASS_COLOR = np.array([255, 255, 255], dtype=np.uint8)

TRAIN_COORDINATES_COLOR = np.array([0, 0, 0], dtype=np.uint8)  # this is black
VALIDATION_COORDINATES_COLOR = np.array([0, 128, 255], dtype=np.uint8)  # this is orange in BGR


def get_most_likely_class_image_array(
    class_probability_map: np.array, class_colors: List, prediction_threshold: float = None
):
    """Plot an image with the most likely class on each pixel."""
    assert (
        len(class_probability_map.shape) == 3
    ), "unexpected class probability array shape"

    number_of_classes = class_probability_map.shape[0]
    assert len(class_colors) == number_of_classes, "an incorrect number of colors has been provided"
    class_colormap = [tuple([int(c) for c in t]) for t in class_colors]

    # append the undefined class color
    class_colormap.append(tuple(int(c) for c in UNDEFINED_CLASS_COLOR))

    # A vanishing total probability indicates that the "one-hot" probability was filled with zero -> unknown class.
    total_probability_map = reduce(class_probability_map, "c h w -> h w", sum)
    missing_label_mask = np.isclose(total_probability_map, 0.0)

    classes_map = get_class_index_from_probabilities(class_probability_map, prediction_threshold)

    cannot_predict_mask = classes_map == CANNOT_PREDICT_INDICATOR

    bad_prediction_mask = missing_label_mask | cannot_predict_mask

    # Give bad predictions an artificial label beyond the number classes
    label_map = classes_map.astype(np.int16)
    label_map[bad_prediction_mask] = number_of_classes

    class_colormap = draw_utils.get_cv_colormap_from_class_color_map(class_colormap)
    image_label_map = draw_utils.apply_cv_colormap(label_map, class_colormap)

    return image_label_map


def overlay_image_label_map_with_with_colored_points(image_label_map: np.array, coordinates: np.array, color: np.array):
    """Get image label map overlayed with colored points.

    This method changes its input image_label_map.
    Args:
        image_label_map (np.array): an integer array of dimensions [height, width, channels=3] which is a colored map
            meant to be displayed with openCV.
        coordinates (np.array): an integer array of coordinates of dimension [n, 2], where n is the number
            of coordinates. These are the coordinates of the points where the color will be modified.
        color (np.array): an array of 3 integers that represents an openCV color.

    Returns:
        image_label_map (np.array): the modified input.
    """
    image_label_map[coordinates[:, 0], coordinates[:, 1]] = color
    return image_label_map


def get_most_likely_class_and_lightened_inputs_image_array(
    class_probability_map: np.array, input_labels_map: np.array, class_colors: List, prediction_threshold: float = None
):
    """Plot an image with the most likely class on each pixel."""
    assert len(input_labels_map.shape) == 2, "Unexpected shape"
    image_label_map = get_most_likely_class_image_array(class_probability_map,
                                                        class_colors,
                                                        prediction_threshold)
    input_label_mask = input_labels_map != DONTCARE_SEGM_MASK_LABEL

    # lighten the colors for known labels.
    image_label_map[input_label_mask, :] = (
        image_label_map[input_label_mask, :].astype(np.float32) / 2
    )

    return image_label_map


def create_input_label_map(ground_truth_label_map, training_coordinates):
    """Create height x width tensor with ground truth labels."""
    height, width = ground_truth_label_map.shape
    input_label_map = np.ones([height, width], dtype=int) * DONTCARE_SEGM_MASK_LABEL
    for (i, j) in training_coordinates:
        input_label_map[i, j] = ground_truth_label_map[i, j]

    return input_label_map


def get_ground_truth_probability_map(
    ground_truth_labels_map: np.ndarray, number_of_classes: int
):
    """Get probability map as stochastic arrays from ground truth integer labels."""
    # Dump the "missing" class label into a new index beyond the normal classes
    augmented_label_map = np.where(
        ground_truth_labels_map == DONTCARE_SEGM_MASK_LABEL,
        number_of_classes,
        ground_truth_labels_map,
    )

    labels_map = torch.from_numpy(augmented_label_map).to(torch.long)
    augmented_ground_truth_one_hot = F.one_hot(
        labels_map, num_classes=number_of_classes + 1
    )
    # get rid of the one-hot corresponding to the missing labels
    ground_truth_one_hot = augmented_ground_truth_one_hot[:, :, :-1]
    probability_ground_truth_map = rearrange(
        ground_truth_one_hot.numpy(), "h w c -> c h w"
    )
    return probability_ground_truth_map
