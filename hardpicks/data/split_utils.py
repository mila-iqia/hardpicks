"""Generic utility functions module used by splitters."""

import typing

import numpy as np


def get_test_ids(
    all_ids: typing.Sequence[typing.Any],
    fraction_in_testing_set: float,
    random_number_generator: np.random.Generator,
) -> typing.Sequence[typing.Any]:
    """Returns the list of sample indices (ids) to be used for testing; this list is never empty."""
    assert len(all_ids) > 0, "nothing to split!"
    number_of_test_ids = get_number_of_testing_elements_at_least_one(
        len(all_ids), fraction_in_testing_set,
    )
    assert number_of_test_ids <= len(all_ids)
    test_ids = random_number_generator.choice(
        all_ids, number_of_test_ids, replace=False,
    )
    return test_ids


def get_number_of_testing_elements_at_least_one(
    number_of_elements: int,
    testing_fraction: float,
) -> int:
    """Returns the number of elements to withhold for testing, which is always at least one."""
    assert number_of_elements > 0
    assert 0 < testing_fraction <= 1
    raw_number_of_testing_elements = np.round(
        testing_fraction * number_of_elements
    ).astype(int)
    number_of_testing_elements = max(1, raw_number_of_testing_elements)
    return number_of_testing_elements
