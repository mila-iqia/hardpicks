import numpy as np
import pandas as pd
import pytest
from torch.utils.data import Dataset

import hardpicks.data.fbp.gather_splitter as splitter
import hardpicks.data.split_utils as split_utils

_EMPTY_SET = set()


class FakeShotGatherDataset(Dataset):
    """Class that fakes the behavior of the real ShotGatherDataset class.

    This fake class avoids the need to have an underlying hdf5 file, while exposing
    all the necessary functionalities for the purpose of this test.
    """

    def __init__(self, traces_dataframe):
        self.trace_to_gather_map = self._get_trace_to_group_map(
            traces_dataframe, "gather_id"
        )
        self.trace_to_line_map = self._get_trace_to_group_map(
            traces_dataframe, "line_id"
        )
        self.trace_to_shot_map = self._get_trace_to_group_map(
            traces_dataframe, "shot_id"
        )
        self.gather_to_trace_map = self._get_group_to_trace_map(
            traces_dataframe, "gather_id"
        )
        self.line_to_trace_map = self._get_group_to_trace_map(
            traces_dataframe, "line_id"
        )
        self.shot_to_trace_map = self._get_group_to_trace_map(
            traces_dataframe, "shot_id"
        )
        self._number_of_time_steps = 25
        self._gather_ids = list(np.unique(self.trace_to_gather_map))

    @staticmethod
    def _get_trace_to_group_map(traces_dataframe: pd.DataFrame, group_key: str):
        return traces_dataframe.sort_values(by="trace_id")[group_key].values

    @staticmethod
    def _get_group_to_trace_map(traces_dataframe: pd.DataFrame, group_key: str):
        return {
            key: group_df["trace_id"].values
            for key, group_df in traces_dataframe.groupby(by=group_key)
        }

    def __len__(self):
        return len(self._gather_ids)

    def get_meta_gather(self, gather_id):
        """Return gather metadata."""
        first_trace_id = self.gather_to_trace_map[gather_id][0]
        return dict(
            shot_id=self.trace_to_shot_map[first_trace_id],
            rec_line_id=self.trace_to_line_map[first_trace_id],
            gather_id=gather_id,
        )

    def __getitem__(self, gather_id):
        """Return random data, but always the same for a given index."""
        gather_meta = self.get_meta_gather(gather_id)
        random_number_generator = np.random.default_rng(seed=gather_id)
        number_of_traces = len(self.gather_to_trace_map[gather_id])
        samples = random_number_generator.random(
            [number_of_traces, self._number_of_time_steps]
        )
        return dict(
            **gather_meta,
            samples=samples,
        )


def skip(probability=0.1):
    """Should we skip the next iteration."""
    return np.random.choice([True, False], p=[probability, 1 - probability])


@pytest.fixture(scope="class")
def shot_ids():
    np.random.seed(23423)
    return np.random.randint(1e4, 1e8, 17)


@pytest.fixture(scope="class")
def line_ids():
    np.random.seed(111)
    return np.random.randint(1e4, 1e8, 12)


@pytest.fixture(scope="class")
def traces_dataframe(shot_ids, line_ids):
    np.random.seed(456)
    list_rows = []
    gather_id = 0
    trace_id = 0
    for shot_id in shot_ids:
        for line_id in line_ids:
            if skip():
                continue
            number_of_traces = np.random.randint(10, 100)
            for _ in range(number_of_traces):
                row = {
                    "shot_id": shot_id,
                    "line_id": line_id,
                    "gather_id": gather_id,
                    "trace_id": trace_id,
                }
                list_rows.append(row)
                trace_id += 1
            gather_id += 1
    df = pd.DataFrame(data=list_rows)
    return df


@pytest.fixture(scope="class")
def fake_shot_gather_dataset(traces_dataframe):
    return FakeShotGatherDataset(traces_dataframe)


@pytest.mark.parametrize(
    "fraction_of_shots_in_testing_set,fraction_of_lines_in_testing_set",
    [(0.1, 0.1), (0.2, 0.1), (0.5, 0.5), (0.25, 0.5)],
    scope="class",
)
class TestSplitGatherIds:
    """Test class to organize many simple tests effectively.

    This testing class is inspired by
    https://docs.pytest.org/en/stable/fixture.html#running-multiple-assert-statements-safely
    The goal is to keep each test as unitary as possible (one assert) while organizing the
    setup as cleanly as possible.
    """

    @pytest.fixture(scope="class")
    def split(
        self,
        fake_shot_gather_dataset,
        fraction_of_shots_in_testing_set,
        fraction_of_lines_in_testing_set,
    ):
        """This is the ACT fixture. It generates the data we will test against."""
        random_number_generator = np.random.default_rng(seed=123)
        train_gather_ids, test_gather_ids = splitter.split_gather_ids_into_train_ids_and_test_ids(
            fake_shot_gather_dataset,
            random_number_generator,
            fraction_of_shots_in_testing_set=fraction_of_shots_in_testing_set,
            fraction_of_lines_in_testing_set=fraction_of_lines_in_testing_set,
        )
        return train_gather_ids, test_gather_ids

    @pytest.fixture(scope="class")
    def set_of_all_gather_ids(self, traces_dataframe):
        total_set = set(traces_dataframe["gather_id"].values)
        return total_set

    @pytest.fixture(scope="class")
    def set_of_training_gather_ids(self, split):
        train_gather_ids, _ = split
        training_set = set(train_gather_ids)
        return training_set

    @pytest.fixture(scope="class")
    def training_trace_dataframe(self, traces_dataframe, set_of_training_gather_ids):
        training_df = traces_dataframe[
            traces_dataframe["gather_id"].isin(set_of_training_gather_ids)
        ]
        return training_df

    @pytest.fixture(scope="class")
    def testing_trace_dataframe(self, traces_dataframe, set_of_testing_gather_ids):
        testing_df = traces_dataframe[
            traces_dataframe["gather_id"].isin(set_of_testing_gather_ids)
        ]
        return testing_df

    @pytest.fixture(scope="class")
    def shot_ids_only_in_testing(
        self, training_trace_dataframe, testing_trace_dataframe
    ):
        testing_set = set(testing_trace_dataframe["shot_id"].values)
        training_set = set(training_trace_dataframe["shot_id"].values)
        testing_only_set = testing_set.difference(training_set)
        return list(testing_only_set)

    @pytest.fixture(scope="class")
    def line_ids_only_in_testing(
        self, training_trace_dataframe, testing_trace_dataframe
    ):
        testing_set = set(testing_trace_dataframe["line_id"].values)
        training_set = set(training_trace_dataframe["line_id"].values)
        testing_only_set = testing_set.difference(training_set)
        return list(testing_only_set)

    @pytest.fixture(scope="class")
    def set_of_testing_gather_ids(self, split):
        _, test_gather_ids = split
        testing_set = set(test_gather_ids)
        return testing_set

    def test_training_and_testing_sets_are_distinct(
        self, set_of_training_gather_ids, set_of_testing_gather_ids
    ):
        intersection_set = set_of_training_gather_ids.intersection(
            set_of_testing_gather_ids
        )
        assert (
            intersection_set == _EMPTY_SET
        ), "training and testing gather ids are not distinct"

    def test_training_and_testing_sets_are_complete(
        self,
        set_of_training_gather_ids,
        set_of_testing_gather_ids,
        set_of_all_gather_ids,
    ):
        union_set = set_of_training_gather_ids.union(set_of_testing_gather_ids)
        assert (
            union_set == set_of_all_gather_ids
        ), "training and testing gather ids are not complete"

    def test_groups_of_gathers_are_in_distinct_set(
        self, training_trace_dataframe, testing_trace_dataframe
    ):
        training_set = set(
            (shot, line)
            for shot, line in training_trace_dataframe[["shot_id", "line_id"]].values
        )
        testing_set = set(
            (shot, line)
            for shot, line in testing_trace_dataframe[["shot_id", "line_id"]].values
        )

        intersection_set = training_set.intersection(testing_set)
        assert (
            intersection_set == _EMPTY_SET
        ), "Pairs of (shot id, line id) overlap between training and testing."

    def test_ratio_of_shot_ids_in_testing_dataset(
        self, shot_ids, shot_ids_only_in_testing, fraction_of_shots_in_testing_set
    ):
        expected_size = split_utils.get_number_of_testing_elements_at_least_one(
            len(shot_ids), fraction_of_shots_in_testing_set
        )
        computed_size = len(shot_ids_only_in_testing)
        assert expected_size == computed_size

    def test_ratio_of_line_ids_in_testing_dataset(
        self, line_ids, line_ids_only_in_testing, fraction_of_lines_in_testing_set
    ):
        expected_size = split_utils.get_number_of_testing_elements_at_least_one(
            len(line_ids), fraction_of_lines_in_testing_set
        )
        computed_size = len(line_ids_only_in_testing)
        assert expected_size == computed_size


@pytest.mark.parametrize(
    "fraction_of_shots_in_testing_set,fraction_of_lines_in_testing_set",
    [(0.1, 0.1), (0.2, 0.1), (0.5, 0.5), (0.25, 0.5)],
    scope="class",
)
class TestSplitIntoSubsets:
    @pytest.fixture(scope="class")
    def subsets(
        self,
        fake_shot_gather_dataset,
        fraction_of_shots_in_testing_set,
        fraction_of_lines_in_testing_set,
    ):
        """This is the ACT fixture. It generates the data we will test against."""
        random_number_generator = np.random.default_rng(seed=123)
        train_dataset, test_dataset = splitter.get_train_and_test_sub_datasets(
            fake_shot_gather_dataset,
            random_number_generator,
            fraction_of_shots_in_testing_set=fraction_of_shots_in_testing_set,
            fraction_of_lines_in_testing_set=fraction_of_lines_in_testing_set,
        )
        return train_dataset, test_dataset

    @pytest.fixture
    def train_subset_set_of_gather_indices(self, subsets):
        train_dataset, _ = subsets
        list_train_gather_ids = []
        for sample_dict in train_dataset:
            list_train_gather_ids.append(sample_dict["gather_id"])
        return set(list_train_gather_ids)

    @pytest.fixture
    def test_subset_set_of_gather_indices(self, subsets):
        _, test_dataset = subsets
        list_test_gather_ids = []
        for sample_dict in test_dataset:
            list_test_gather_ids.append(sample_dict["gather_id"])

        return set(list_test_gather_ids)

    def test_lengths_of_subsets(self, fake_shot_gather_dataset, subsets):
        train_dataset, test_dataset = subsets
        assert len(fake_shot_gather_dataset) == len(train_dataset) + len(test_dataset)

    def test_subsets_are_disjoint(
        self, train_subset_set_of_gather_indices, test_subset_set_of_gather_indices
    ):
        intersection_set = train_subset_set_of_gather_indices.intersection(
            test_subset_set_of_gather_indices
        )
        assert intersection_set == _EMPTY_SET
