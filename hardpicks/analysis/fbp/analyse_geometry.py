"""Visualize each site's source and receiver geometry.

This script is meant to be executed. For each site, it extracts the
xy positions of sources and receivers and generates plots to show each
lines of receivers. It creates one pdf report showing various such images
for each site.
"""
from pathlib import Path

import matplotlib.pylab as plt
from tqdm import tqdm

from hardpicks import ROOT_DIR, ANALYSIS_RESULTS_DIR
from hardpicks.analysis.fbp.analysis_plots import (
    plot_source_and_recorders,
    plot_line_by_line,
)
from hardpicks.analysis.fbp.analysis_utils import (
    get_data_from_site_dict,
)
from hardpicks.data.fbp.site_info import get_site_info_array
from hardpicks.utils.file_utils import (
    get_front_page_series,
)

path_to_this_file = Path(__file__).relative_to(ROOT_DIR)

if __name__ == "__main__":

    site_info_array = get_site_info_array()
    for site_dict in tqdm(site_info_array, desc="SITE"):

        fbp_data, site_name, data_file_path = get_data_from_site_dict(site_dict)

        front_page_series = get_front_page_series(data_file_path, path_to_this_file)

        recorder_df = fbp_data.get_recorder_dataframe()
        source_df = fbp_data.get_source_dataframe()

        list_image_paths = []

        fig = plot_source_and_recorders(source_df, recorder_df)
        fig.suptitle(f"Arrangement of recorders and sources: {site_name} site")
        image_path = ANALYSIS_RESULTS_DIR.joinpath("sources_and_receivers.png")
        fig.savefig(image_path)
        plt.close(fig)
        list_image_paths.append(image_path)

        list_figs = plot_line_by_line(recorder_df)
        for i, fig in enumerate(list_figs):
            fig.suptitle(f"{site_name} site")
            image_path = ANALYSIS_RESULTS_DIR.joinpath(f"receiver_line_{i}.png")
            fig.savefig(image_path)
            plt.close(fig)
            list_image_paths.append(image_path)

        # pdf_output_path = ANALYSIS_RESULTS_DIR.joinpath(f"{site_name}_geometry.pdf")
        # write_report_and_cleanup(pdf_output_path, list_image_paths, front_page_series)
