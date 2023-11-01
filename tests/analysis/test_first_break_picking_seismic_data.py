import numpy as np
import pandas as pd

import pytest

from hardpicks.analysis.fbp.first_break_picking_seismic_data import (
    FirstBreakPickingSeismicData,
)


@pytest.mark.parametrize("receiver_id_digit_count", [3, 4])
class TestFirstBreakPickingSeismicData:
    """Test class to organize many simple tests effectively.
    This testing class is inspired by
    https://docs.pytest.org/en/stable/fixture.html#running-multiple-assert-statements-safely
    The goal is to keep each test as unitary as possible (one assert) while organizing the
    setup as cleanly as possible.
    """

    @pytest.fixture()
    def first_break_picking_seismic_data(self, hdf5_path, receiver_id_digit_count):
        return FirstBreakPickingSeismicData(hdf5_path, receiver_id_digit_count)

    def test_get_recorder_line_number(
        self, fake_data, first_break_picking_seismic_data
    ):
        computed_line_numbers = first_break_picking_seismic_data._get_recorder_line_number(
            fake_data["REC_PEG"].values
        )
        expected_line_numbers = fake_data["line_number"].values
        np.testing.assert_equal(computed_line_numbers, expected_line_numbers)

    def test_get_gather_indices(self, fake_data, first_break_picking_seismic_data):
        groups = fake_data.groupby(by=["SHOT_PEG", "line_number"])

        for (shot_peg, line_number), group_df in groups:
            expected_indices = group_df.index.values
            computed_indices = first_break_picking_seismic_data.get_gather_indices(
                shot_peg, line_number
            )

            np.testing.assert_equal(expected_indices, computed_indices)

    def test_get_recorder_dataframe(self, fake_data, first_break_picking_seismic_data):
        expected_recorder_df = (
            fake_data[["REC_PEG", "line_number", "REC_X", "REC_Y", "REC_HT"]]
            .drop_duplicates()
            .set_index("REC_PEG")
        ).rename(columns={"REC_X": "x", "REC_Y": "y", "REC_HT": "z"})
        expected_recorder_df.index.name = "peg"

        computed_recorder_df = first_break_picking_seismic_data.get_recorder_dataframe()

        pd.testing.assert_frame_equal(
            expected_recorder_df,
            computed_recorder_df,
            check_dtype=False,
            check_index_type=False,
        )

    def test_get_source_dataframe(self, fake_data, first_break_picking_seismic_data):
        expected_source_df = (
            fake_data[["SHOT_PEG", "SOURCE_X", "SOURCE_Y", "SOURCE_HT"]]
            .drop_duplicates()
            .set_index("SHOT_PEG")
        ).rename(columns={"SOURCE_X": "x", "SOURCE_Y": "y", "SOURCE_HT": "z"})
        expected_source_df.index.name = "peg"

        computed_source_df = first_break_picking_seismic_data.get_source_dataframe()

        pd.testing.assert_frame_equal(
            expected_source_df,
            computed_source_df,
            check_dtype=False,
            check_index_type=False,
        )

    def test_get_measurement_dataframe(
        self, fake_data, first_break_picking_seismic_data
    ):
        expected_df = fake_data[
            ["SHOT_PEG", "REC_PEG", "line_number", "REC_X", "REC_Y"]
        ].rename(
            columns={
                "SHOT_PEG": "shot_peg",
                "REC_PEG": "record_peg",
                "REC_X": "x",
                "REC_Y": "y",
            }
        )
        computed_df = first_break_picking_seismic_data.get_measurement_dataframe()

        pd.testing.assert_frame_equal(expected_df, computed_df, check_dtype=False)

    def test_get_one_dimensional_dataset(
        self, fake_data, first_break_picking_seismic_data
    ):
        attribute_name = "SPARE1"
        computed_array = first_break_picking_seismic_data.get_one_dimensional_dataset(
            attribute_name
        )

        expected_array = fake_data[attribute_name].values

        np.testing.assert_equal(computed_array, expected_array)

    def test_get_single_parameter(self, fake_specs, first_break_picking_seismic_data):
        expected_samp_rate = fake_specs.samp_rate
        computed_samp_rate = first_break_picking_seismic_data._get_single_parameter(
            "SAMP_RATE"
        )

        assert expected_samp_rate == computed_samp_rate

    def test_get_first_breaks_array(
        self, fake_specs, fake_data, first_break_picking_seismic_data
    ):
        computed_first_breaks_in_milliseconds = (
            first_break_picking_seismic_data._get_first_breaks_array()
        )

        sample_rate_in_microseconds = fake_specs.samp_rate

        sample_rate_in_milliseconds = sample_rate_in_microseconds / 1000

        expected_first_break_in_milliseconds = (
            fake_data["SPARE1"].values / sample_rate_in_milliseconds
        )

        np.testing.assert_equal(
            expected_first_break_in_milliseconds, computed_first_breaks_in_milliseconds
        )


def test_get_index_of_value_in_array():
    value = 1.234
    value_array = np.array([value, 2, 3, 4, value, value])
    expected_indices = np.array([0, 4, 5])
    computed_indices = FirstBreakPickingSeismicData._get_index_of_value_in_array(
        value_array, value
    )
    np.testing.assert_equal(expected_indices, computed_indices)


def test_check_array_as_single_unique_value():
    FirstBreakPickingSeismicData._check_array_as_single_unique_value(
        np.array([1, 1, 1])
    )
    with pytest.raises(AssertionError):
        FirstBreakPickingSeismicData._check_array_as_single_unique_value(
            np.array([1, 2, 3])
        )
