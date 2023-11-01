import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hardpicks.data.fbp.receiver_location_corrections import (
    HALFMILE_BAD_RECORDER_PEGS_SWAP_PAIRS,
    preprocess_halfmile_dataset,
    _get_bad_recorder_pegs_swap_dictionary,
)
from tests.fake_fbp_data_utilities import (
    create_site_receiver_geometry,
    create_site_sources_geometry,
    create_fake_site_data_from_parameters,
    FakeSiteSpecifications,
    create_fake_hdf5_dataset,
    assert_two_hdf5_contain_same_data, HDF5_INT_KEYS, HDF5_GEO_KEYS, HDF5_2D_FLOAT_KEYS,
)


def create_fake_site_data_width_swapped_halfmile_recorder_pegs(
    fake_site_specifications: FakeSiteSpecifications,
):
    rng = np.random.default_rng(fake_site_specifications.random_seed)
    source_geometry_df = create_site_sources_geometry(fake_site_specifications, rng)

    receiver_geometry_df = create_site_receiver_geometry(fake_site_specifications, rng)

    good_order_receiver_geometry_df = receiver_geometry_df.copy()
    bad_order_receiver_geometry_df = receiver_geometry_df.copy()

    number_of_swap_pairs = len(HALFMILE_BAD_RECORDER_PEGS_SWAP_PAIRS)

    all_indices = np.arange(len(receiver_geometry_df["REC_PEG"]))

    swap_indices_pairs = rng.choice(
        all_indices, size=2 * number_of_swap_pairs, replace=False
    ).reshape((number_of_swap_pairs, 2))

    for (rec_peg1, rec_peg2), (i1, i2) in zip(
        HALFMILE_BAD_RECORDER_PEGS_SWAP_PAIRS, swap_indices_pairs
    ):
        good_order_receiver_geometry_df["REC_PEG"][i1] = rec_peg1
        good_order_receiver_geometry_df["REC_PEG"][i2] = rec_peg2

        bad_order_receiver_geometry_df["REC_PEG"][i1] = rec_peg2
        bad_order_receiver_geometry_df["REC_PEG"][i2] = rec_peg1

    number_of_time_samples = fake_site_specifications.number_of_time_samples

    fresh_rng1 = np.random.default_rng(fake_site_specifications.random_seed)
    fresh_rng2 = np.random.default_rng(fake_site_specifications.random_seed)

    first_break_field_name = 'SPARE1'
    good_fake_data = create_fake_site_data_from_parameters(
        good_order_receiver_geometry_df,
        source_geometry_df,
        number_of_time_samples,
        fake_site_specifications.shot_id_is_corrupted,
        fake_site_specifications.samp_rate,
        first_break_field_name,
        fresh_rng1,
    )

    bad_fake_data = create_fake_site_data_from_parameters(
        bad_order_receiver_geometry_df,
        source_geometry_df,
        number_of_time_samples,
        fake_site_specifications.shot_id_is_corrupted,
        fake_site_specifications.samp_rate,
        first_break_field_name,
        fresh_rng2,
    )

    return bad_fake_data, good_fake_data


@pytest.fixture()
def fake_raw_and_processed_data(fake_specs):
    return create_fake_site_data_width_swapped_halfmile_recorder_pegs(fake_specs)


@pytest.fixture()
def paths_to_hdf5(fake_raw_and_processed_data):
    raw_data, processed_data = fake_raw_and_processed_data
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("Writing fake data to temp directory")

        input_hdf5_path = Path(tmp_dir_str).joinpath("raw.hdf5")
        output_hdf5_path = Path(tmp_dir_str).joinpath("computed.hdf5")
        expected_hdf5_path = Path(tmp_dir_str).joinpath("expected.hdf5")

        create_fake_hdf5_dataset(raw_data, input_hdf5_path)
        create_fake_hdf5_dataset(processed_data, expected_hdf5_path)

        yield input_hdf5_path, output_hdf5_path, expected_hdf5_path


def test_get_bad_recorder_pegs_swap_dictionary():
    bar_recorder_pegs_swap_pairs = [(1, 2), (3, 4)]
    expected_dict = {1: 2, 2: 1, 3: 4, 4: 3}
    computed_dict = _get_bad_recorder_pegs_swap_dictionary(bar_recorder_pegs_swap_pairs)

    assert expected_dict == computed_dict


@pytest.mark.parametrize("receiver_id_digit_count", [4])
def test_preprocess_halfmile_dataset(paths_to_hdf5):
    (
        path_to_raw_dataset,
        path_to_output_processed_dataset,
        path_to_expected_dataset,
    ) = paths_to_hdf5
    preprocess_halfmile_dataset(
        str(path_to_raw_dataset), str(path_to_output_processed_dataset)
    )

    assert os.path.exists(
        path_to_output_processed_dataset
    ), "output file does not exist"

    hdf_keys = HDF5_INT_KEYS + HDF5_GEO_KEYS + HDF5_2D_FLOAT_KEYS + ['SPARE1']

    assert_two_hdf5_contain_same_data(path_to_output_processed_dataset, path_to_expected_dataset, hdf_keys)
