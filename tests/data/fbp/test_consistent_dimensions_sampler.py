import numpy as np
import pytest

from hardpicks.data.fbp.consistent_dimensions_sampler import (
    PowerTwoDimensionsGroupBatchSampler,
)
from hardpicks.data.fbp.gather_wrappers import ShotLineGatherConcatDataset
from hardpicks.utils.processing_utils import get_nearest_power_of_two


class FakeDataset:
    def __init__(self, list_data_dictionaries):
        self.list_data_dictionaries = list_data_dictionaries

    def __len__(self):
        return len(self.list_data_dictionaries)

    def __getitem__(self, gather_id):
        data_dictionary = self.get_meta_gather(gather_id)
        data_and_sample = {**data_dictionary, "sample": [1, 2, 3]}
        return data_and_sample

    def get_meta_gather(self, gather_id):
        return self.list_data_dictionaries[gather_id]


def get_data_dictionaries(number_of_examples, origin):
    trace_dims = np.random.randint(1, 250, number_of_examples)
    sample_dims = np.random.randint(1, 250, number_of_examples)

    return [
        dict(origin=origin, gather_id=i, trace_count=t, sample_count=s)
        for i, (t, s) in enumerate(zip(trace_dims, sample_dims))
    ]


@pytest.fixture()
def dataset1():
    np.random.seed(34232)
    return FakeDataset(get_data_dictionaries(number_of_examples=1000, origin="A"))


@pytest.fixture()
def dataset2():
    np.random.seed(132)
    return FakeDataset(get_data_dictionaries(number_of_examples=500, origin="B"))


@pytest.fixture()
def concatenated_dataset(dataset1, dataset2):
    return ShotLineGatherConcatDataset([dataset1, dataset2])


@pytest.mark.parametrize("batch_size, minimum_power", [(1, 0), (2, 1), (4, 2), (8, 3)])
def test_power_two_dimensions_group_batch_sampler(concatenated_dataset, batch_size, minimum_power):

    batch_sampler = PowerTwoDimensionsGroupBatchSampler(
        concatenated_dataset, batch_size=batch_size, minimum_power=minimum_power
    )

    all_indices = []
    for batch_indices in batch_sampler:
        assert len(batch_indices) <= batch_size
        all_indices.extend(batch_indices)

        list_keys = []
        for idx in batch_indices:
            data_dict = concatenated_dataset[idx]
            t2 = max(get_nearest_power_of_two(data_dict["trace_count"]), 2**minimum_power)
            s2 = max(get_nearest_power_of_two(data_dict["sample_count"]), 2**minimum_power)
            list_keys.append((t2, s2))

        assert len(set(list_keys)) == 1

    np.testing.assert_array_equal(
        np.sort(all_indices), np.arange(len(concatenated_dataset))
    )
