import numpy as np
import pytest

import hardpicks.data.split_utils as split_utils


@pytest.mark.parametrize(
    "number_of_elements,testing_fraction,expected_number_of_elements",
    [(100, 0.1, 10), (10, 0.01, 1), (10, 0.26, 3)],
)
def test_get_number_of_testing_elements_at_least_one__parametrized(
        number_of_elements, testing_fraction, expected_number_of_elements
):
    computed_number_of_testing_elements = split_utils.get_number_of_testing_elements_at_least_one(
        number_of_elements, testing_fraction
    )
    assert computed_number_of_testing_elements == expected_number_of_elements


def test_get_number_of_testing_elements_at_least_one__exceptions():

    with pytest.raises(AssertionError):
        _ = split_utils.get_number_of_testing_elements_at_least_one(0, 0.5)

    with pytest.raises(AssertionError):
        _ = split_utils.get_number_of_testing_elements_at_least_one(5, 0)


def test_get_test_ids():

    for _ in range(1000):
        idxs_count = np.random.randint(1, 200)
        orig_idxs = [i for i in np.random.randint(0, np.iinfo(np.int32).max, (idxs_count, ))]
        test_frac = max(np.random.rand(), 0.00001)
        test_idxs = split_utils.get_test_ids(
            all_ids=orig_idxs,
            fraction_in_testing_set=test_frac,
            random_number_generator=np.random.default_rng(),
        )
        assert len(test_idxs) <= len(orig_idxs)
        assert all([idx in orig_idxs for idx in test_idxs])
        assert len(np.unique(test_idxs)) == len(test_idxs)
        assert len(test_idxs) >= 1
        assert np.floor(test_frac * idxs_count) <= len(test_idxs) <= np.ceil(test_frac * idxs_count)
