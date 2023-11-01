from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from hardpicks.analysis.fbp.analysis_utils import (
    get_normalized_displacements,
    get_base_ax_index,
    compute_basic_statistics,
    get_deduplicated_displacement_dataframe,
    is_one_to_one,
)
from hardpicks.data.fbp.gather_cleaner import ShotLineGatherCleaner
from hardpicks.data.fbp.gather_parser import create_shot_line_gather_dataset


@pytest.fixture
def positions_and_normalized_displacements():

    np.random.seed(0)
    number_of_values = 100

    radii = np.random.rand(number_of_values)
    angles = 2 * np.pi * np.random.rand(number_of_values)

    x0, y0 = np.random.rand(2)

    list_normalized_dx = np.concatenate([[np.NaN], np.cos(angles)])
    list_normalized_dy = np.concatenate([[np.NaN], np.sin(angles)])

    list_dx = radii * list_normalized_dx[1:]
    list_dy = radii * list_normalized_dy[1:]

    list_x = np.concatenate([[x0], x0 + np.cumsum(list_dx)])
    list_y = np.concatenate([[y0], y0 + np.cumsum(list_dy)])

    return list_x, list_y, list_normalized_dx, list_normalized_dy


@pytest.fixture
def measurements_df_and_expected_displacement_df():

    np.random.seed(23423)
    number_of_points = 10

    list_x = np.random.rand(number_of_points)
    list_y = np.random.rand(number_of_points)

    ones = np.ones(number_of_points, dtype=int)

    shot_pegs = np.concatenate([123 * ones, 456 * ones])
    line_numbers = 111 * ones

    measurements_df = pd.DataFrame(
        data={
            "shot_peg": shot_pegs,
            "line_number": np.concatenate([line_numbers, line_numbers]),
            "x": np.concatenate([list_x, list_x]),
            "y": np.concatenate([list_y, list_y]),
        }
    )

    list_normalized_dx, list_normalized_dy = get_normalized_displacements(
        list_x, list_y
    )

    deduplicated_df = pd.DataFrame(
        data={
            "line_number": line_numbers,
            "x": list_x,
            "y": list_y,
            "dx": list_normalized_dx,
            "dy": list_normalized_dy,
        }
    )

    return measurements_df, deduplicated_df


def test_get_normalized_displacements(positions_and_normalized_displacements):

    list_x, list_y, list_dx, list_dy = positions_and_normalized_displacements

    list_computed_dx, list_computed_dy = get_normalized_displacements(list_x, list_y)

    np.testing.assert_allclose(list_computed_dx, list_dx)
    np.testing.assert_allclose(list_computed_dy, list_dy)


@pytest.mark.parametrize(
    "number_of_plots,base_index", [(16, (4, 4)), (8, (2, 4)), (4, (1, 4)), (10, (3, 4))]
)
def test_get_base_ax_index(number_of_plots, base_index):
    assert base_index == get_base_ax_index(number_of_plots, number_of_plots_per_row=4)


@pytest.fixture
def clean_dataset(hdf5_path, receiver_id_digit_count):
    raw_dataset = create_shot_line_gather_dataset(
        hdf5_path=hdf5_path,
        site_name='test',
        receiver_id_digit_count=receiver_id_digit_count,
        first_break_field_name='SPARE1',
        provide_offset_dists=True,
        convert_to_fp16=False,
        convert_to_int16=False,
    )
    clean_dataset = ShotLineGatherCleaner(dataset=raw_dataset,
                                          auto_invalidate_outlier_picks=False,
                                          )
    return clean_dataset


@pytest.mark.parametrize("receiver_id_digit_count", [3, 4], scope="session")
def test_compute_basic_statistics(fake_data, clean_dataset):
    computed_stats_df = compute_basic_statistics(clean_dataset)

    # Explicitly cast to int64 as different platform seem to do different things, breaking the test
    for column_name in computed_stats_df.columns:
        computed_stats_df[column_name] = computed_stats_df[column_name].astype(np.int64)

    groups = fake_data.groupby(by=["SHOT_PEG", "line_number"])

    list_rows = []
    for (shot_peg, line_number), group_df in groups:
        row = OrderedDict()
        row["shot peg"] = shot_peg
        row["line number"] = line_number
        row["gather size"] = len(group_df)
        row["bad picks"] = np.sum(group_df["SPARE1"] <= 0)
        row["dead traces"] = np.sum(group_df["sample"].apply(np.linalg.norm) < 1e-8)

        list_rows.append(row)

    expected_stats_df = pd.DataFrame(list_rows)

    pd.testing.assert_frame_equal(expected_stats_df, computed_stats_df)


def test_get_deduplicated_displacement_dataframe(
    measurements_df_and_expected_displacement_df,
):
    (
        measurements_df,
        expected_displacement_df,
    ) = measurements_df_and_expected_displacement_df
    computed_displacement_df = get_deduplicated_displacement_dataframe(measurements_df)

    pd.testing.assert_frame_equal(expected_displacement_df, computed_displacement_df)


@pytest.mark.parametrize(
    "list_i, list_j,expected_results",
    [
        (np.array([1, 1, 1, 2, 3]), np.array([4, 4, 4, 5, 6]), True),
        (np.array([1, 1, 1, 2, 3]), np.array([8, 4, 4, 5, 6]), False),
        (np.array([1, 2, 2, 3, 3]), np.array([9, 1, 1, 6, 6]), True),
    ],
)
def test_is_one_to_one(list_i, list_j, expected_results):
    computed_results = is_one_to_one(list_i, list_j)
    assert computed_results == expected_results
