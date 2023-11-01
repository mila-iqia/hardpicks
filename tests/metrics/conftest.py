import pytest

import numpy as np
import pandas as pd

from hardpicks.metrics.classification_utils import (
    CANNOT_PREDICT_INDICATOR,
)
from hardpicks.models.constants import DONTCARE_SEGM_MASK_LABEL


@pytest.fixture()
def ground_truth_column_name():
    return "some_random_column_name"


@pytest.fixture()
def prediction_column_name():
    return "some_other_name"


@pytest.fixture()
def predicted_label_probability_column_name():
    return "probability_label_some_name"


@pytest.fixture
def number_of_points():
    return 321


@pytest.fixture
def classes():
    return list(range(3))


@pytest.fixture
def ground_truth_classes(classes, number_of_points):
    np.random.seed(12312)
    augmented_classes = [DONTCARE_SEGM_MASK_LABEL] + classes
    ground_truth_classes = np.random.choice(augmented_classes, number_of_points)
    return ground_truth_classes


@pytest.fixture
def predicted_classes(classes, number_of_points):
    np.random.seed(3455234)
    augmented_classes = [CANNOT_PREDICT_INDICATOR] + classes
    predicted_classes = np.random.choice(augmented_classes, number_of_points)
    return predicted_classes


@pytest.fixture
def predicted_probabilities(number_of_points):
    np.random.seed(234232)
    return np.random.random(number_of_points)


@pytest.fixture
def dataframe(
    ground_truth_classes,
    predicted_classes,
    predicted_probabilities,
    ground_truth_column_name,
    prediction_column_name,
    predicted_label_probability_column_name,
):
    df = pd.DataFrame(
        data={
            ground_truth_column_name: ground_truth_classes,
            prediction_column_name: predicted_classes,
            predicted_label_probability_column_name: predicted_probabilities,
        }
    )
    return df
