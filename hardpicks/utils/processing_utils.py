import math
from typing import List, Any

import numpy as np


def normalize_one_dimensional_array(value_array: np.array):
    """Normalize array by subtracting mean and divide by standard deviation."""
    array_std = np.std(value_array)

    if array_std < 1.0e-6:
        std = 1.0
    else:
        std = array_std

    return (value_array - np.mean(value_array)) / std


def get_nearest_power_of_two(value: int) -> int:
    """Compute the closest integer of the form 2^n larger than value."""
    return int(2 ** (math.ceil(math.log(value, 2))))


def concatenate_lists(list_of_lists: List[List[Any]]) -> List[Any]:
    """Concatenate lists.

    Args:
        list_of_lists (List):  list of elements which are lists.

    Returns:
        flattened_list: the concatenation of all the lists in the list
    """
    combined_list = []
    for sub_list in list_of_lists:
        combined_list.extend(sub_list)
    return combined_list
