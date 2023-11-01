"""Extract basic stats for each site.

This script is meant to be executed. It iterates over all sites to extract
basic statistics (number of line gathers, number of shots, etc) and
it outputs the results as a table in a pdf document.
"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hardpicks import ANALYSIS_RESULTS_DIR
from hardpicks.analysis.fbp.analysis_plots import plot_basic_stats
from hardpicks.analysis.fbp.analysis_utils import (
    compute_basic_statistics,
)
from hardpicks.data.fbp.gather_cleaner import ShotLineGatherCleaner
from hardpicks.data.fbp.gather_parser import create_shot_line_gather_dataset
from hardpicks.data.fbp.site_info import get_site_info_array

output_path = ANALYSIS_RESULTS_DIR.joinpath("basic_stats_sites_summary.pdf")

main_sites = [
    'Sudbury',
    'Halfmile',
    # 'Matagami',
    'Brunswick',
    'Lalor',
]

if __name__ == "__main__":

    list_rows = []
    site_info_array = get_site_info_array()
    for site_info in tqdm(site_info_array, desc="SITE"):
        site_name = site_info['site_name']
        if site_name not in main_sites:
            continue

        raw_dataset = create_shot_line_gather_dataset(
            hdf5_path=site_info['processed_hdf5_path'],
            site_name=site_info['site_name'],
            receiver_id_digit_count=site_info['receiver_id_digit_count'],
            first_break_field_name=site_info['first_break_field_name'],
            provide_offset_dists=True,
        )
        clean_dataset = ShotLineGatherCleaner(dataset=raw_dataset,
                                              auto_invalidate_outlier_picks=False,
                                              )
        number_of_all_gather_ids = len(np.unique(raw_dataset.trace_to_gather_map))

        stats_df = compute_basic_statistics(clean_dataset)
        number_of_clean_valid_gather_ids = len(clean_dataset.valid_gather_ids)
        assert len(stats_df) == number_of_clean_valid_gather_ids, "numbers don't match"

        fig = plot_basic_stats(stats_df)
        fig.savefig(ANALYSIS_RESULTS_DIR.joinpath(f"{site_name}_basic_stats.png"))
        plt.close(fig)

        row = {
            "site": site_name,
            "total line gathers": number_of_all_gather_ids,
            "valid line gathers": len(stats_df),
            "shots": len(np.unique(stats_df["shot peg"].values)),
            "lines": len(np.unique(stats_df["line number"].values)),
            "traces": stats_df["gather size"].sum(),
            "avg bad picks": np.round(stats_df["bad picks"].mean(), 1),
            "avg dead traces": np.round(stats_df["dead traces"].mean(), 1),
            "avg size": np.round(stats_df["gather size"].mean(), 1),
        }

        list_rows.append(row)

    summary_df = pd.DataFrame(list_rows)
    print(summary_df)

    # create_sites_summary_report(summary_df, output_path)
