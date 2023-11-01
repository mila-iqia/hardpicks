from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd


good_shot_peg_per_site = {
    "Halfmile": "SHOT_PEG",
    "Brunswick": "SHOT_PEG",
    "Lalor": "SHOT_PEG",
    "Sudbury": "SHOTID",
    "Matagami": "SHOT_PEG",
    "Kevitsa": "SHOT_PEG",
}


class FirstBreakPickingSeismicData:
    """This class wraps around a hdf5 dataset of seismic traces and first break picks.

    It is designed to be subclassed for specific sites.
    """

    @property
    def receiver_peg_divider(self):
        """Factor that separates line number from peg number."""
        return self._receiver_peg_divider

    @property
    def shot_peg_key(self):
        """Name of the field containing the shot pegs."""
        return self._shot_peg_key

    @property
    def first_break_pick_key(self):
        """Name of the field containing the first break times."""
        return self._first_break_pick_key

    def __init__(
        self,
        path_to_hdf5_file: Path,
        receiver_id_digit_count: int,
        shot_peg_key: str = "SHOT_PEG",
        first_break_pick_key: str = "SPARE1",
    ):
        """Initialize class and create useful arrays."""
        self._receiver_peg_divider = 10 ** receiver_id_digit_count
        self._shot_peg_key = shot_peg_key
        self._first_break_pick_key = first_break_pick_key

        self.data_file = h5py.File(path_to_hdf5_file, "r")
        self.base_group = self.data_file["/TRACE_DATA/DEFAULT/"]
        self.normalized_first_breaks = self._get_first_breaks_array()
        self.raw_traces = self.base_group["data_array"]

        self.record_pegs = self.get_one_dimensional_dataset("REC_PEG")
        self.shot_pegs = self.get_one_dimensional_dataset(self.shot_peg_key)

        self.record_line_numbers = self._get_recorder_line_number(self.record_pegs)

        self.first_breaks_in_milliseconds = self.get_one_dimensional_dataset(
            first_break_pick_key
        )

        # the sample rate is stored in microseconds.
        self.sample_rate_milliseconds = self._get_single_parameter("SAMP_RATE") / 1000
        self.time_in_milliseconds = (
            np.arange(self.raw_traces.shape[1]) * self.sample_rate_milliseconds
        )

    def get_gather_indices(self, shot_peg: int, line_number: int):
        """Get indices for given shot peg and line number."""
        shot_indices = self._get_index_of_value_in_array(self.shot_pegs, shot_peg)
        line_indices = self._get_index_of_value_in_array(
            self.record_line_numbers, line_number
        )
        gather_indices = np.intersect1d(shot_indices, line_indices)
        return gather_indices

    @staticmethod
    def _get_index_of_value_in_array(value_array: np.array, value: Union[float, int]):
        return np.where(value_array == value)[0]

    def _check_shot_pegs_and_shotid_are_same(self):
        """Validate that shotid is the same thing as shot peg."""
        shot_ids = self.get_one_dimensional_dataset("SHOTID")

        assert (
            np.linalg.norm(self.shot_pegs - shot_ids) == 0.0
        ), "shotid is not the same as shot peg"

    def _get_site_dataframe(self, peg_tag: str, x_tag: str, y_tag: str, z_tag: str):
        pegs = self.get_one_dimensional_dataset(peg_tag)
        data = dict(
            peg=pegs,
            line_number=self._get_recorder_line_number(pegs),
            x=self.get_one_dimensional_dataset(x_tag),
            y=self.get_one_dimensional_dataset(y_tag),
            z=self.get_one_dimensional_dataset(z_tag),
        )
        raw_df = pd.DataFrame(data=data)

        # the z data is not precise
        df = raw_df.drop_duplicates(subset=["peg", "line_number", "x", "y"])
        peg_values = df["peg"].values
        assert len(peg_values) == len(np.unique(peg_values)), "pegs are not unique!"
        return df.set_index("peg")

    def get_recorder_dataframe(self):
        """Get pandas dataframe for recorder properties."""
        return self._get_site_dataframe("REC_PEG", "REC_X", "REC_Y", "REC_HT")

    def get_source_dataframe(self):
        """Get pandas dataframe for source properties."""
        src_df = self._get_site_dataframe(
            self.shot_peg_key, "SOURCE_X", "SOURCE_Y", "SOURCE_HT"
        )
        del src_df["line_number"]
        return src_df

    def get_measurement_dataframe(self):
        """Get pandas dataframe with all shot and recording data."""
        record_pegs = self.get_one_dimensional_dataset("REC_PEG")
        data = dict(
            shot_peg=self.get_one_dimensional_dataset(self.shot_peg_key),
            record_peg=record_pegs,
            line_number=self._get_recorder_line_number(record_pegs),
            x=self.get_one_dimensional_dataset("REC_X"),
            y=self.get_one_dimensional_dataset("REC_Y"),
        )
        df = pd.DataFrame(data=data)
        return df

    def _get_recorder_line_number(self, record_pegs: np.array):
        """The recorder line number is represented by the first few digits of the record peg."""
        return np.trunc(record_pegs / self.receiver_peg_divider).astype(int)

    @staticmethod
    def _check_array_as_single_unique_value(value_array: np.array):
        """Check that array contain single unique value."""
        assert (
            len(np.unique(value_array.flatten())) == 1
        ), "the array contains more than one unique value"

    def get_one_dimensional_dataset(self, attribute_name: str):
        """Extract attribute_name property from the hdf5 file."""
        return self.base_group[attribute_name][:].flatten()

    def _get_single_parameter(self, attribute_name: str):
        values = self.get_one_dimensional_dataset(attribute_name)
        self._check_array_as_single_unique_value(values)
        return values[0]

    def _get_first_breaks_array(self):
        sample_rate = self._get_single_parameter("SAMP_RATE")
        dt = sample_rate / 1000.0
        raw_first_breaks = self.get_one_dimensional_dataset(self.first_break_pick_key)
        normalized_first_breaks = raw_first_breaks / dt
        return normalized_first_breaks
