import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from hardpicks import ROOT_DIR, ANALYSIS_RESULTS_DIR
from hardpicks.analysis.fbp.analysis_plots import (
    plot_basic_stats,
    plot_source_and_recorders,
    plot_line_by_line,
    plot_normalized_displacements,
    get_shot_figure,
)
from hardpicks.analysis.fbp.analysis_utils import (
    compute_basic_statistics,
    get_data_from_site_dict,
)
from hardpicks.data.fbp.site_info import get_site_info_array
from hardpicks.utils.file_utils import (
    get_front_page_series,
)

number_of_shot_gathers = 10

path_to_this_script = Path(__file__).relative_to(ROOT_DIR)

if __name__ == "__main__":

    site_info_array = get_site_info_array()
    for site_dict in tqdm(site_info_array, desc="SITE"):

        fbp_data, site_name, data_file_path = get_data_from_site_dict(site_dict)

        front_page_series = get_front_page_series(data_file_path, path_to_this_script)

        recorder_df = fbp_data.get_recorder_dataframe()
        source_df = fbp_data.get_source_dataframe()
        measurement_df = fbp_data.get_measurement_dataframe()
        stats_df = compute_basic_statistics(fbp_data)

        list_image_paths = []

        logging.info("Plotting geometry")

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

        logging.info("Plotting basic statistics")

        fig = plot_basic_stats(stats_df)
        image_path = ANALYSIS_RESULTS_DIR.joinpath("basic_stats.png")
        fig.savefig(image_path)
        plt.close(fig)
        list_image_paths.append(image_path)

        logging.info("Plotting normalized displacements")

        fig = plot_normalized_displacements(measurement_df)
        image_path = ANALYSIS_RESULTS_DIR.joinpath(
            "normalized_displacements_along_receiver_lines.png"
        )
        fig.savefig(image_path)
        plt.close(fig)
        list_image_paths.append(image_path)

        logging.info("Plotting example gathers")

        unique_shot_pegs = np.unique(fbp_data.shot_pegs)
        unique_lines_numbers = np.unique(fbp_data.record_line_numbers)

        list_shot_pegs = np.random.choice(unique_shot_pegs, number_of_shot_gathers)

        for shot_peg in tqdm(list_shot_pegs, desc="SHOT PEG"):
            for line_number in unique_lines_numbers:

                gather_indices = fbp_data.get_gather_indices(shot_peg, line_number)
                if len(gather_indices) == 0:
                    continue

                fig = get_shot_figure(fbp_data, shot_peg, line_number)

                filename = f"traces_shot_{shot_peg}_line_{line_number}.png"
                output_path = ANALYSIS_RESULTS_DIR.joinpath(filename)
                fig.savefig(output_path)
                plt.close(fig)
                list_image_paths.append(output_path)

        # pdf_output_path = ANALYSIS_RESULTS_DIR.joinpath(f"{site_name}_report.pdf")
        # write_report_and_cleanup(pdf_output_path, list_image_paths, front_page_series)
