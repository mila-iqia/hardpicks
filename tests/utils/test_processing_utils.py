import pytest
import numpy as np

from hardpicks.utils.processing_utils import (
    normalize_one_dimensional_array,
    get_nearest_power_of_two,
    concatenate_lists,
)

values = np.array([1.0, 2.0, 3.0])
normalized_values = (values - np.mean(values)) / np.std(values)
zero_values = np.array([0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "values,expected_normalized_values",
    [(values, normalized_values), (zero_values, zero_values)],
)
def test_normalize_one_dimensional_array(values, expected_normalized_values):
    computed_normalized_values = normalize_one_dimensional_array(values)
    np.testing.assert_array_equal(
        computed_normalized_values, expected_normalized_values
    )


@pytest.mark.parametrize("input,expected_value", [(1, 1), (5, 8), (12, 16)])
def test_get_nearest_power_of_two(input, expected_value):
    computed_value = get_nearest_power_of_two(input)
    assert computed_value == expected_value


def test_concatenate_lists():

    p1 = (2, 4)
    p2 = (5, 1)
    p3 = (4, 4)
    p4 = (1, 1)
    p5 = (0, 0)
    p6 = (0, 23)
    p7 = (12, 17)

    expected_list = [p1, p2, p3, p4, p5, p6, p7]
    l1 = [p1, p2]
    l2 = [p3, p4]
    l3 = [p5, p6, p7]
    list_of_lists = [l1, l2, l3]
    computed_list = concatenate_lists(list_of_lists)

    assert computed_list == expected_list
