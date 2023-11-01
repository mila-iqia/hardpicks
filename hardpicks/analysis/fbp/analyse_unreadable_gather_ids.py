"""Identify bad gather ids.

This script is meant to be executed. It iterates over all datasets to identify
ids that cannot be loaded (if any) and documents the error that is thrown.
This is then written to a lightweight csv file for inspection.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from hardpicks import ANALYSIS_RESULTS_DIR
from hardpicks.analysis.logging_utils import setup_analysis_logger
from hardpicks.data.fbp.gather_cleaner import ShotLineGatherCleaner
from hardpicks.data.fbp.gather_parser import (
    create_shot_line_gather_dataset,
)
from hardpicks.data.fbp.site_info import get_site_info_array

logger = logging.getLogger(__name__)
setup_analysis_logger()

if __name__ == "__main__":

    list_rows = []

    site_info_array = get_site_info_array()
    for site_info in tqdm(site_info_array, desc="SITE", file=sys.stderr):

        site_name = site_info["site_name"]

        shot_line_gather_dataset = create_shot_line_gather_dataset(
            hdf5_path=site_info["processed_hdf5_path"],
            site_name=site_name,
            receiver_id_digit_count=site_info["receiver_id_digit_count"],
            first_break_field_name=site_info["first_break_field_name"],
            convert_to_fp16=True,
            convert_to_int16=True,
            preload_trace_data=False,
            cache_trace_metadata=False,
            provide_offset_dists=True,
        )

        clean_site_dataset = ShotLineGatherCleaner(
            shot_line_gather_dataset, auto_fill_missing_picks=False
        )
        gather_ids = np.arange(len(clean_site_dataset))

        for gather_id in tqdm(gather_ids, "IDX", file=sys.stderr):
            try:
                datum = clean_site_dataset[gather_id]
            except Exception as e:
                logger.error(f"site {site_name}: Cannot load gather id {gather_id}. Error: {e.args[0]}")

                row = dict(site=site_name, gather_id=gather_id, error=e.args[0])
                list_rows.append(row)

    error_df = pd.DataFrame(list_rows)

    output_path = ANALYSIS_RESULTS_DIR.joinpath("bad_gather_ids_report.csv")
    error_df.to_csv(output_path, index=False, sep='\t')
