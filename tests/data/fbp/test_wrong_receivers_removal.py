import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hardpicks.data.fbp.wrong_receivers_removal import (
    get_good_receiver_indices,
    preprocess_matagami_dataset,
)
from tests.fake_fbp_data_utilities import (
    create_fake_hdf5_dataset,
    assert_two_hdf5_contain_same_data, HDF5_INT_KEYS, HDF5_2D_FLOAT_KEYS, HDF5_GEO_KEYS, )


@pytest.fixture()
def reduced_fake_data_and_bad_receiver_pegs(fake_data):
    np.random.seed(2323)

    unique_pegs = fake_data["REC_PEG"].unique()

    bad_receiver_pegs = np.random.choice(
        unique_pegs, int(0.25 * len(unique_pegs)), replace=False
    )

    reduced_fake_data = fake_data[
        ~fake_data["REC_PEG"].isin(bad_receiver_pegs)
    ].reset_index(drop=True)

    return reduced_fake_data, bad_receiver_pegs


@pytest.fixture()
def reduced_fake_data(reduced_fake_data_and_bad_receiver_pegs):
    return reduced_fake_data_and_bad_receiver_pegs[0]


@pytest.fixture()
def bad_receiver_pegs(reduced_fake_data_and_bad_receiver_pegs):
    return reduced_fake_data_and_bad_receiver_pegs[1]


@pytest.fixture()
def paths_to_hdf5(fake_data, reduced_fake_data):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("Writing fake data to temp directory")

        input_hdf5_path = Path(tmp_dir_str).joinpath("raw.hdf5")
        output_hdf5_path = Path(tmp_dir_str).joinpath("computed.hdf5")
        expected_hdf5_path = Path(tmp_dir_str).joinpath("expected.hdf5")

        create_fake_hdf5_dataset(fake_data, input_hdf5_path)
        create_fake_hdf5_dataset(reduced_fake_data, expected_hdf5_path)

        yield input_hdf5_path, output_hdf5_path, expected_hdf5_path


@pytest.fixture
def unique_peg_values():
    np.random.seed(123)
    number_of_unique_pegs = 10

    return np.random.randint(1e4, 1e8, number_of_unique_pegs)


@pytest.fixture
def unique_bad_receiver_pegs(unique_peg_values):
    np.random.seed(456)
    return np.random.choice(unique_peg_values, 3, replace=False)


@pytest.fixture
def all_receiver_pegs_and_good_peg_indices(unique_peg_values, unique_bad_receiver_pegs):

    np.random.seed(789)

    set_of_good_pegs = set(unique_peg_values).difference(set(unique_bad_receiver_pegs))
    unique_good_pegs = len(set_of_good_pegs)

    number_of_pegs = 1000
    number_of_bad_pegs = 50

    all_receiver_pegs = np.random.choice(unique_good_pegs, number_of_pegs)

    bad_pegs = np.random.choice(unique_bad_receiver_pegs, number_of_bad_pegs)
    bad_peg_indices = np.random.choice(
        np.arange(number_of_pegs), number_of_bad_pegs, replace=False
    )
    all_receiver_pegs[bad_peg_indices] = bad_pegs

    good_peg_indices = np.sort(
        list(set(np.arange(number_of_pegs)).difference(set(bad_peg_indices)))
    )

    return all_receiver_pegs, good_peg_indices


@pytest.fixture
def all_receiver_pegs(all_receiver_pegs_and_good_peg_indices):
    return all_receiver_pegs_and_good_peg_indices[0]


@pytest.fixture
def good_peg_indices(all_receiver_pegs_and_good_peg_indices):
    return all_receiver_pegs_and_good_peg_indices[1]


def test_get_good_receiver_indices(
    all_receiver_pegs, good_peg_indices, unique_bad_receiver_pegs
):
    computed_good_peg_indices = get_good_receiver_indices(
        all_receiver_pegs, unique_bad_receiver_pegs
    )

    np.testing.assert_array_equal(computed_good_peg_indices, good_peg_indices)


@pytest.mark.parametrize("receiver_id_digit_count", [4])
def test_preprocess_matagami_dataset(paths_to_hdf5, bad_receiver_pegs):
    (
        path_to_raw_dataset,
        path_to_output_processed_dataset,
        path_to_expected_dataset,
    ) = paths_to_hdf5

    hdf_keys = HDF5_INT_KEYS + HDF5_GEO_KEYS + HDF5_2D_FLOAT_KEYS + ['SPARE1']

    preprocess_matagami_dataset(
        str(path_to_raw_dataset),
        str(path_to_output_processed_dataset),
        hdf_keys,
        bad_receiver_pegs,
    )

    assert_two_hdf5_contain_same_data(
        path_to_output_processed_dataset, path_to_expected_dataset, hdf_keys
    )
