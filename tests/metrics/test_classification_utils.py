import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hardpicks.metrics.classification_utils import (
    get_class_index_from_probabilities, CANNOT_PREDICT_INDICATOR,
)


@pytest.fixture
def number_of_classes():
    return 12


@pytest.fixture
def probabilities(number_of_classes, number_of_dimensions):
    np.random.seed(23423)
    dimensions = np.random.randint(2, 10, number_of_dimensions)
    scores = np.random.rand(number_of_classes, *dimensions)
    preds = F.softmax(torch.from_numpy(scores), dim=0).numpy()
    return preds


@pytest.fixture
def expected_classes(probabilities, prediction_threshold):

    number_of_classes = probabilities.shape[0]
    spatial_shape = probabilities.shape[1:]

    classes = -999 * np.ones(spatial_shape, dtype=np.int16)

    list_ranges = [list(range(d)) for d in spatial_shape]

    for idx in itertools.product(*list_ranges):
        index = (slice(0, number_of_classes), *idx)
        highest_probability_class_index = np.argmax(probabilities[index])

        index = (highest_probability_class_index, *idx)
        highest_probability = probabilities[index]

        if prediction_threshold is None or highest_probability >= prediction_threshold:
            classes[idx] = highest_probability_class_index
        else:
            classes[idx] = CANNOT_PREDICT_INDICATOR

    return classes


@pytest.mark.parametrize("number_of_dimensions", [2, 3, 4, 5])
@pytest.mark.parametrize("prediction_threshold", [None, 0.0, 0.1, 0.5, 0.99])
def test_get_class_index_from_probabilities(
    probabilities, expected_classes, number_of_dimensions, prediction_threshold
):

    computed_classes = get_class_index_from_probabilities(
        probabilities, prediction_threshold=prediction_threshold
    )

    assert len(computed_classes.shape) == number_of_dimensions
    assert computed_classes.shape == probabilities.shape[1:]

    np.testing.assert_equal(computed_classes, expected_classes)
