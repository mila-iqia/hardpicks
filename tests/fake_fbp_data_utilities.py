from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from hardpicks.data.fbp.constants import (
    BAD_FIRST_BREAK_PICK_INDEX,
)

_INT_TYPE_STR = "int32"
_FLOAT_TYPE_STR = "float32"

HDF5_INT_KEYS = ["REC_PEG", "SHOT_PEG", "SHOTID", "SAMP_RATE"]

HDF5_SPARE_KEYS = ["SPARE1", "SPARE2", "SPARE3", "SPARE4"]

HDF5_GEO_KEYS = [
    "REC_X",
    "REC_Y",
    "REC_HT",
    "SOURCE_X",
    "SOURCE_Y",
    "SOURCE_HT",
]

HDF5_FLOAT_KEYS = HDF5_SPARE_KEYS + HDF5_GEO_KEYS

HDF5_2D_FLOAT_KEYS = ["data_array"]

HDF5_KEYS = HDF5_INT_KEYS + HDF5_FLOAT_KEYS + HDF5_2D_FLOAT_KEYS

IntegerRange = namedtuple("IntegerRange", ["min", "max"])

FakeSiteSpecifications = namedtuple(
    "FakeSiteSpecifications",
    [
        "site_name",
        "random_seed",
        "samp_rate",
        "receiver_id_digit_count",
        "first_break_field_name",
        "number_of_time_samples",
        "number_of_shots",
        "number_of_lines",
        "shot_id_is_corrupted",
        "range_of_receivers_per_line",
    ],
)


def get_random_position(rng):
    return 1.0 * rng.integers(1e2, 5e3)


def create_site_receiver_geometry(
    fake_site_specifications: FakeSiteSpecifications, rng
):

    line_numbers = rng.integers(1e4, 9e4, fake_site_specifications.number_of_lines)
    receiver_range = fake_site_specifications.range_of_receivers_per_line

    receiver_line_prefix = 10 ** fake_site_specifications.receiver_id_digit_count

    list_rows = []
    for line_number in line_numbers:
        number_of_receivers = rng.integers(receiver_range.min, receiver_range.max)
        for _ in np.arange(number_of_receivers):
            prefix = line_number * receiver_line_prefix
            record_peg = prefix + rng.integers(1, receiver_line_prefix - 1)

            row = {
                "REC_PEG": record_peg,
                "line_number": line_number,
                "REC_X": get_random_position(rng),
                "REC_Y": get_random_position(rng),
                "REC_HT": get_random_position(rng),
            }
            list_rows.append(row)

    return pd.DataFrame(list_rows)


def create_site_sources_geometry(fake_site_specifications: FakeSiteSpecifications, rng):
    shot_pegs = rng.integers(1e8, 1e9, fake_site_specifications.number_of_shots)

    list_rows = []

    for shot_peg in shot_pegs:
        row = {
            "SHOT_PEG": shot_peg,
            "SOURCE_X": get_random_position(rng),
            "SOURCE_Y": get_random_position(rng),
            "SOURCE_HT": get_random_position(rng),
        }
        list_rows.append(row)

    return pd.DataFrame(list_rows)


def create_fake_site_data(fake_site_specifications: FakeSiteSpecifications):
    """This function creates a pandas dataframe filled with random data to describe a site."""
    rng = np.random.default_rng(fake_site_specifications.random_seed)

    receiver_geometry_df = create_site_receiver_geometry(fake_site_specifications, rng)
    source_geometry_df = create_site_sources_geometry(fake_site_specifications, rng)

    fake_data = create_fake_site_data_from_parameters(
        receiver_geometry_df,
        source_geometry_df,
        fake_site_specifications.number_of_time_samples,
        fake_site_specifications.shot_id_is_corrupted,
        fake_site_specifications.samp_rate,
        fake_site_specifications.first_break_field_name,
        rng,
    )

    return fake_data


def create_fake_site_data_from_parameters(
    receiver_geometry_df,
    source_geometry_df,
    number_of_time_samples,
    shot_id_is_corrupted,
    samp_rate,
    first_break_field_name,
    rng,
):
    """This function creates a pandas dataframe filled with random data to describe a site."""

    fbp_choices = np.concatenate(
        [
            [BAD_FIRST_BREAK_PICK_INDEX - 1, BAD_FIRST_BREAK_PICK_INDEX],
            np.arange(number_of_time_samples),
        ]
    )

    receiver_line_groups = receiver_geometry_df.groupby("line_number")
    trace_exists_dict = {}
    for shot_peg in source_geometry_df["SHOT_PEG"].values:
        for line_number, line_df in receiver_line_groups:
            list_recorder_pegs = line_df["REC_PEG"].values
            line_exists = rng.choice([True, False])
            for record_peg in list_recorder_pegs:
                record_exists = rng.choice([True, False])
                trace_exists_dict[(shot_peg, line_number, record_peg)] = (
                    line_exists and record_exists
                )

    list_rows = []

    for _, receiver_row in receiver_geometry_df.iterrows():
        line_number = int(receiver_row["line_number"])
        record_peg = int(receiver_row["REC_PEG"])
        rec_x = receiver_row["REC_X"]
        rec_y = receiver_row["REC_Y"]
        rec_z = receiver_row["REC_HT"]

        for _, source_row in source_geometry_df.iterrows():
            shot_peg = int(source_row["SHOT_PEG"])
            src_x = source_row["SOURCE_X"]
            src_y = source_row["SOURCE_Y"]
            src_z = source_row["SOURCE_HT"]

            if not trace_exists_dict[(shot_peg, line_number, record_peg)]:
                continue

            trace_is_dead = rng.choice([True, False])

            if trace_is_dead:
                sample = np.zeros(number_of_time_samples, dtype=np.float32)
            else:
                sample = rng.random(number_of_time_samples, dtype=np.float32)

            fbp = rng.choice(fbp_choices)

            if shot_id_is_corrupted:
                shot_id = rng.integers(0, 10)
            else:
                shot_id = shot_peg

            row = {
                'SAMP_RATE': samp_rate,
                'SHOT_PEG': shot_peg,
                'SHOTID': shot_id,
                'REC_PEG': record_peg,
                'line_number': line_number,
                'REC_X': rec_x,
                'REC_Y': rec_y,
                'REC_HT': rec_z,
                'SOURCE_X': src_x,
                'SOURCE_Y': src_y,
                'SOURCE_HT': src_z,
                first_break_field_name: fbp,
                'sample': sample,
            }
            list_rows.append(row)

    return pd.DataFrame(list_rows)


def create_fake_hdf5_dataset(fake_data: pd.DataFrame, output_hdf5_path: Path):

    hdf5_file = h5py.File(output_hdf5_path, "w")

    number_of_traces = len(fake_data)
    number_of_time_samples = len(fake_data["sample"][0])

    base_group = hdf5_file.create_group("/TRACE_DATA/DEFAULT/")

    for list_keys, type in zip(
        [HDF5_INT_KEYS, HDF5_FLOAT_KEYS], [_INT_TYPE_STR, _FLOAT_TYPE_STR]
    ):
        for key in list_keys:
            if key not in fake_data.columns:
                continue
            pointer = base_group.create_dataset(
                key, (number_of_traces, 1), dtype=type, track_times=False
            )
            pointer[...] = fake_data[key].values.reshape(-1, 1)

    pointer = base_group.create_dataset(
        "data_array",
        (number_of_traces, number_of_time_samples),
        dtype=_FLOAT_TYPE_STR,
        track_times=False,
    )
    pointer[...] = np.stack(fake_data["sample"].values)

    ones_array = np.ones([number_of_traces, 1]).astype(int)

    for extra_key, factor in zip(
        ["SAMP_NUM", "COORD_SCALE", "HT_SCALE"], [number_of_time_samples, 1, 1]
    ):
        pointer = base_group.create_dataset(
            extra_key, (number_of_traces, 1), dtype=_INT_TYPE_STR, track_times=False
        )
        pointer[...] = factor * ones_array
    hdf5_file.close()


def assert_two_hdf5_contain_same_data(hdf_path1, hdf_path2, list_keys):
    data_files = []
    base_groups = []
    for hdf_path in [hdf_path1, hdf_path2]:
        data_file = h5py.File(hdf_path, mode="r")
        data_files.append(data_file)
        base_groups.append(data_file["/TRACE_DATA/DEFAULT/"])

    for key in list_keys:
        array1 = base_groups[0][key][:]
        array2 = base_groups[1][key][:]
        np.testing.assert_array_equal(array1, array2)

    for data_file in data_files:
        data_file.close()
