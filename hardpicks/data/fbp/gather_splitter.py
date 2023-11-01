import logging
import typing

import numpy as np

import hardpicks.data.fbp.gather_parser as gather_parser
import hardpicks.data.fbp.gather_wrappers as gather_wrappers
import hardpicks.data.split_utils as split_utils

logger = logging.getLogger(__name__)


def get_train_and_test_sub_datasets(
    shot_line_gather_dataset: gather_parser.ShotLineGatherDatasetBase,
    random_number_generator: np.random.Generator,
    fraction_of_shots_in_testing_set: float = 0.1,
    fraction_of_lines_in_testing_set: float = 0.1,
    ignore_line_ids_if_unique: bool = False,
) -> typing.Tuple[gather_wrappers.ShotLineGatherSubset, gather_wrappers.ShotLineGatherSubset]:
    """Method to extract training and testing pytorch Subset objects for the original dataset."""
    train_gather_ids, test_gather_ids = split_gather_ids_into_train_ids_and_test_ids(
        shot_line_gather_dataset,
        random_number_generator,
        fraction_of_shots_in_testing_set,
        fraction_of_lines_in_testing_set,
        ignore_line_ids_if_unique,
    )
    return (
        gather_wrappers.ShotLineGatherSubset(shot_line_gather_dataset, train_gather_ids),
        gather_wrappers.ShotLineGatherSubset(shot_line_gather_dataset, test_gather_ids),
    )


def split_gather_ids_into_train_ids_and_test_ids(
    shot_line_gather_dataset: gather_parser.ShotLineGatherDatasetBase,
    random_number_generator: np.random.Generator,
    fraction_of_shots_in_testing_set: float = 0.1,
    fraction_of_lines_in_testing_set: float = 0.1,
    ignore_line_ids_if_unique: bool = False,
) -> typing.Tuple[typing.List[int], typing.List[int]]:
    """Get the gather ids for the training and testing sets for a given gather parser/wrapper object.

    Since we can't guarantee we'll have access to the real parser object here, we can't use its maps,
    and instead have to rely on the metadata from loaded gathers.
    """
    # we'll do an initial loop to gather (no pun) all shot/line/gather ids
    shot_map, line_map = {}, {}
    for gather_idx in range(len(shot_line_gather_dataset)):
        gather_meta = shot_line_gather_dataset.get_meta_gather(gather_idx)
        shot_id, line_id = gather_meta["shot_id"], gather_meta["rec_line_id"]
        # note: we're using "SHOT LINE GATHERS", meaning one shot can produce multiple gathers
        #       (i.e. one shot with N rec lines = N shot line gathers)
        if shot_id not in shot_map:
            shot_map[shot_id] = []
        shot_map[shot_id].append(gather_idx)
        if line_id not in line_map:
            line_map[line_id] = []
        line_map[line_id].append(gather_idx)

    test_gather_ids = []

    # first, determine which gathers will be reserved for the test set based on shot IDs...
    test_shot_ids = split_utils.get_test_ids(
        list(shot_map.keys()), fraction_of_shots_in_testing_set, random_number_generator,
    )
    for shot_id in test_shot_ids:
        test_gather_ids.extend(shot_map[shot_id])

    # if possible, determine which gathers will be reserved for the test set based on line IDs...
    if not ignore_line_ids_if_unique or len(line_map) > 1:
        test_line_ids = split_utils.get_test_ids(
            list(line_map.keys()), fraction_of_lines_in_testing_set, random_number_generator,
        )
        for line_id in test_line_ids:
            test_gather_ids.extend(line_map[line_id])
    else:
        logger.warning("ignoring line-id-based splitting for dataset due to unique receiver line")

    test_gather_ids = np.unique(test_gather_ids)
    train_gather_ids = np.delete(np.arange(len(shot_line_gather_dataset)), test_gather_ids)

    assert len(test_gather_ids) + len(train_gather_ids) == len(shot_line_gather_dataset)

    return list(train_gather_ids), list(test_gather_ids)
