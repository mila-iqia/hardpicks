import logging
from abc import abstractmethod

import torch
from torch.utils.data import SubsetRandomSampler, BatchSampler
from tqdm import tqdm

from hardpicks.utils.processing_utils import get_nearest_power_of_two

logger = logging.getLogger(__name__)


class GroupBatchSampler:
    """Create batches within groups.

    This base class implements a batch sampler that will create batches within
    data sub-groupings. The creation of the actual groups is controlled by method
    _get_groups_map, which is abstract: this is thus deferred to child classes.
    """

    def __init__(self, data_source, batch_size: int, generator=None, **grouping_kwargs):
        """Initialize class.

        Args:
            data_source: the dataset that will be used to create the groups.
            batch_size: the maximum batch size.
            generator: the torch random generator.
            grouping_kwargs: keyword arguments to be passed to the grouping method.
        """
        self.generator = generator
        self.batch_size = batch_size

        # the groups_map object is a dictionary of the form group_keys:[list of indices in group]
        logger.info("Computing group maps...")
        groups_map = self._get_groups_map(data_source, **grouping_kwargs)
        logger.info("Done Computing group maps.")

        self.list_batch_sub_samplers = self._get_list_batch_sub_samplers(groups_map)

        self.sampler_lengths = [
            len(batch_sub_sampler) for batch_sub_sampler in self.list_batch_sub_samplers
        ]

        self.ordered_sampler_indices = self._get_ordered_sampler_indices(
            self.sampler_lengths
        )

    def __len__(self):
        """Return total number of batches."""
        return sum(self.sampler_lengths)

    def __iter__(self):
        """Create iterator that returns the next batch."""
        # In order to sample all batches from all sub_batch_samplers,
        # we'll call each sub_batch_sampler exactly the number of batches it
        # contains. We'll shuffle the order of sub_batch_sampler calls.
        number_of_sampler_iterations = len(self.ordered_sampler_indices)
        shuffled_integers = torch.randperm(
            number_of_sampler_iterations, generator=self.generator
        )

        list_sample_iterators = [
            iter(batch_sub_sampler)
            for batch_sub_sampler in self.list_batch_sub_samplers
        ]

        for i in shuffled_integers:
            sampler_index = self.ordered_sampler_indices[i]
            sampler_iterator = list_sample_iterators[sampler_index]
            yield next(sampler_iterator)

    @staticmethod
    def _get_ordered_sampler_indices(list_sampler_lengths):
        ordered_sampler_indices = []
        for sampler_index, sampler_length in enumerate(list_sampler_lengths):
            ordered_sampler_indices.extend(sampler_length * [sampler_index])

        return ordered_sampler_indices

    def _get_list_batch_sub_samplers(self, groups_map):

        list_batch_sub_samplers = []
        for _, indices in groups_map.items():
            sampler = SubsetRandomSampler(indices, generator=self.generator)
            batch_sub_sampler = BatchSampler(sampler, self.batch_size, drop_last=False)
            list_batch_sub_samplers.append(batch_sub_sampler)

        return list_batch_sub_samplers

    @abstractmethod
    def _get_groups_map(self, data_source, **grouping_kwargs):
        pass


class PowerTwoDimensionsGroupBatchSampler(GroupBatchSampler):
    """Concrete implementation of the GroupBatchSampler.

    The grouping is by power of 2 padding of the (width, height) of samples.
    """

    def _get_groups_map(self, dataset, minimum_power: int = 6):

        assert hasattr(
            dataset, "get_meta_gather"
        ), "The dataset should have the get_meta_gather method."

        dimension_groups_map = dict()

        minimum_power_of_two = 2 ** minimum_power

        number_of_elements = len(dataset)

        for idx in tqdm(range(number_of_elements), "Grouping batch sampler"):
            meta_gather = dataset.get_meta_gather(idx)

            tp2 = max(
                get_nearest_power_of_two(meta_gather["trace_count"]),
                minimum_power_of_two,
            )
            sp2 = max(
                get_nearest_power_of_two(meta_gather["sample_count"]),
                minimum_power_of_two,
            )

            key = (tp2, sp2)
            if key in dimension_groups_map:
                dimension_groups_map[key].append(idx)
            else:
                logger.info(f"New grouping key: {key}")
                dimension_groups_map[key] = [idx]

        return dimension_groups_map
