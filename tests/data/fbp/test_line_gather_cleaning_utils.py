import numpy as np
import pytest

from hardpicks.data.fbp.line_gather_cleaning_utils import (
    get_fitted_distances,
    fit_distance_with_time,
    unpack_data_dict, get_good_times_and_distances, get_global_linear_fit_and_all_errors,
)


@pytest.fixture()
def v0():
    np.random.seed(12423)
    v0 = 10 * np.random.random()
    return v0


@pytest.fixture()
def t0():
    np.random.seed(123)
    t0 = 100 * np.random.random()
    return t0


@pytest.fixture()
def number_of_samples():
    return 1000


@pytest.fixture()
def number_of_gathers():
    return 20


@pytest.fixture()
def times(number_of_samples):
    np.random.seed(422)
    times = 1000 * np.random.random(number_of_samples)
    return times


@pytest.fixture()
def distances(v0, t0, times):
    distances = v0 * (times - t0)
    return distances


@pytest.fixture()
def bad_mask(number_of_samples):
    np.random.seed(34523)
    mask = np.random.choice([True, False], number_of_samples, p=[0.8, 0.2])
    return mask


@pytest.fixture
def data_dict(distances, times):
    np.random.seed(123)
    zeros = np.zeros_like(distances)
    data = dict(
        first_break_timestamps=times,
        offset_distances=np.vstack([distances, zeros, zeros]).T,
        bad_first_breaks_mask=np.random.choice(
            [True, False], len(distances), p=[0.9, 0.1]
        ),
    )
    return data


class FakeDataset:

    def __init__(self, list_data):
        self.list_data = list_data

    def __len__(self):
        return len(self.list_data)

    def get_meta_gather(self, gather_id):
        return self.list_data[gather_id]


@pytest.fixture
def fake_dataset(times, distances, bad_mask, number_of_gathers):
    split_times = np.split(times, number_of_gathers)
    split_distances = np.split(distances, number_of_gathers)
    split_masks = np.split(bad_mask, number_of_gathers)

    list_data = []
    for gather_times, gather_distances, gather_masks in zip(split_times, split_distances, split_masks):
        data = dict(
            first_break_timestamps=gather_times,
            offset_distances=gather_distances.reshape(-1, 1),
            bad_first_breaks_mask=gather_masks)
        list_data.append(data)
    return FakeDataset(list_data)


def test_get_fitted_distances(v0, t0, times, distances):
    computed_distances = get_fitted_distances(v0, t0, times)
    np.testing.assert_array_equal(distances, computed_distances)


def test_fit_distance_with_time(v0, t0, times, distances):
    computed_v0, computed_t0 = fit_distance_with_time(distances, times)

    assert np.isclose(computed_v0, v0)
    assert np.isclose(computed_t0, t0)


def test_unpack_data_dict(distances, times, data_dict):

    computed_distances, computed_times, bad_mask = unpack_data_dict(data_dict)

    np.testing.assert_equal(computed_distances, distances)
    np.testing.assert_equal(computed_times, times)
    np.testing.assert_equal(bad_mask, data_dict["bad_first_breaks_mask"])


def test_get_good_times_and_distances(fake_dataset, times, distances, bad_mask):
    expected_good_distances = distances[~bad_mask]
    expected_good_times = times[~bad_mask]

    computed_good_distances, computed_good_times = get_good_times_and_distances(fake_dataset)

    np.testing.assert_array_equal(expected_good_distances, computed_good_distances)
    np.testing.assert_array_equal(expected_good_times, computed_good_times)


def test_get_global_linear_fit_and_all_errors(fake_dataset, v0, t0):
    computed_v0, computed_t0, computed_all_absolute_errors = get_global_linear_fit_and_all_errors(fake_dataset)

    assert np.isclose(computed_v0, v0)
    assert np.isclose(computed_t0, t0)
    assert np.allclose(computed_all_absolute_errors, 0.)
