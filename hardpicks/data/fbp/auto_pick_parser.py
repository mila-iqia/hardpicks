import re
from io import StringIO
import numpy as np

import pandas as pd
import torch
from tqdm import tqdm

from hardpicks.data.fbp.constants import BAD_OR_PADDED_ELEMENT_ID, BAD_FIRST_BREAK_PICK_INDEX
from hardpicks.data.fbp.trace_parser import RawTraceDataset


class AutoPickParser:
    """Class to parse and manipulate the autopick data."""

    def __init__(
        self, site_name: str, shot_identifier_key: str, autopicks_file_path: str, sample_rate_in_milliseconds: int
    ):
        """Initialize the class."""
        assert shot_identifier_key == 'shot_id' or shot_identifier_key == 'shot_number',\
            "shot_identifier_key should be shot_id or shot_number"
        self.shot_identifier_key = shot_identifier_key

        self.site_name = site_name
        self.autopicks_df = self.get_autopicks_dataframe(autopicks_file_path)
        self.autopicks_series = self._get_autopicks_series(
            self.autopicks_df, sample_rate_in_milliseconds
        )

    def get_raw_preds(self, batch):
        """Extract the needed autopicks from the pic file."""
        list_predicted_picks = []
        for shot_id, rec_ids in zip(batch["shot_id"].numpy(), batch["rec_ids"].numpy()):

            predicted_picks = []
            for rec_id in rec_ids:
                if rec_id == BAD_OR_PADDED_ELEMENT_ID:
                    autopick = BAD_FIRST_BREAK_PICK_INDEX
                else:
                    autopick = self._get_autopick_value(shot_id, rec_id)
                predicted_picks.append(autopick)
            list_predicted_picks.append(predicted_picks)

        raw_preds = torch.from_numpy(np.array(list_predicted_picks))
        return raw_preds

    def _get_autopick_value(self, shot_id, rec_id):
        index = (shot_id, rec_id)
        if index in self.autopicks_series.index:
            return self.autopicks_series.loc[index]
        else:
            return BAD_FIRST_BREAK_PICK_INDEX

    @classmethod
    def _get_dataframe_from_block(cls, block):
        """Parse a block of picks."""
        header_regex = r"ENSEMBLE NO : *(?P<shot_number>\d+)  *(?P<shot_id>\d+)"
        header_line = block[0]
        regex_match = re.match(header_regex, header_line)
        shot_number = int(regex_match.group("shot_number"))
        shot_id = int(regex_match.group("shot_id"))

        data_string = "".join(block[2:])
        df = pd.read_csv(
            StringIO(data_string),
            delim_whitespace=True,
            names=["Trace", "Time", "receiver_id"],
        )
        df["shot_number"] = shot_number
        df["shot_id"] = shot_id
        return df

    @classmethod
    def get_autopicks_dataframe(cls, picks_file_path):
        """Parse the full autopicks file."""
        block_start_key = "ENSEMBLE NO :"

        with open(picks_file_path, "r") as f:
            lines = f.readlines()

        list_start_block_indices = []

        for i, line in enumerate(lines):
            if block_start_key in line:
                list_start_block_indices.append(i)

        list_end_block_indices = list_start_block_indices[1:] + [len(lines)]

        list_blocks = []
        for start_index, end_index in zip(
            list_start_block_indices, list_end_block_indices
        ):
            list_blocks.append(lines[start_index:end_index])

        list_df = []

        for block in tqdm(list_blocks):
            df = cls._get_dataframe_from_block(block)
            list_df.append(df)

        autopicks_df = pd.concat(list_df)
        return autopicks_df

    def _get_autopicks_series(self, autopicks_df, sample_rate_in_milliseconds):
        fbp_times_in_milliseconds = autopicks_df["Time"].values

        first_break_labels = RawTraceDataset._get_first_break_indices(
            fbp_times_in_milliseconds, sample_rate_in_milliseconds
        )
        autopicks_df["first_break_labels"] = first_break_labels

        autopicks_series = autopicks_df.set_index([self.shot_identifier_key, "receiver_id"])[
            "first_break_labels"
        ]
        return autopicks_series
