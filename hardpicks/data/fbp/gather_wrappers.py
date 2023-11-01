"""This module contains some wrappers around torch dataset interfaces for shot line gathers."""

import bisect
import typing

import torch.utils.data

import hardpicks.data.fbp.gather_parser as gather_parser


class ShotLineGatherConcatDataset(
    torch.utils.data.ConcatDataset,
    gather_parser.ShotLineGatherDatasetBase,
):
    """Wrapper around `ConcatDataset` that exposes `get_meta_gather` in the underlying datasets."""

    def __init__(
        self,
        datasets: typing.Iterable[gather_parser.ShotLineGatherDatasetBase],
    ) -> None:
        """Initialize class; validate that needed method is present."""
        for dataset in datasets:
            assert hasattr(
                dataset, "get_meta_gather"
            ), "The datasets to be concatenated should have the get_meta_gather method."
        super(ShotLineGatherConcatDataset, self).__init__(datasets)

    def get_meta_gather(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Split indices the same way as __getitem__, but call get_meta_gather instead."""
        if gather_id < 0:
            if -gather_id > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            gather_id = len(self) + gather_id
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, gather_id)
        if dataset_idx == 0:
            sample_idx = gather_id
        else:
            sample_idx = gather_id - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_meta_gather(sample_idx)

    def get_sample_weights(self) -> torch.Tensor:
        """Returns an array of sample weights used for dataset-level rebalancing."""
        # note: these are NOT normalized in any way, so careful how they're used!
        return torch.cat([torch.ones(len(d)) / (len(d) / len(self)) for d in self.datasets])


class ShotLineGatherSubset(
    torch.utils.data.Subset,
    gather_parser.ShotLineGatherDatasetBase,
):
    """Wrapper around `Subset` that exposes `get_meta_gather` in the underlying dataset."""

    def __init__(
        self,
        dataset: gather_parser.ShotLineGatherDatasetBase,
        indices: typing.Sequence[int],
    ) -> None:
        """Initialize class; validate that needed method is present."""
        assert hasattr(
            dataset, "get_meta_gather"
        ), "the dataset to be subset should have the get_meta_gather method."
        super(ShotLineGatherSubset, self).__init__(dataset=dataset, indices=indices)

    def get_meta_gather(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Split indices the same way as __getitem__, but call get_meta_gather instead."""
        if gather_id < 0:
            if -gather_id > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            gather_id = len(self) + gather_id
        return self.dataset.get_meta_gather(self.indices[gather_id])
