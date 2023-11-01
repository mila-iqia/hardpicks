import logging
import os
import pickle

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


from hardpicks.analysis.fbp.report.path_constants import (
    style_path,
    pickles_directory,
    output_directory,
)
from hardpicks import FBP_BAD_GATHERS_DIR
from hardpicks.analysis.fbp.analysis_plots import plot_basic_stats
from hardpicks.analysis.fbp.analysis_utils import (
    compute_basic_statistics,
)
from hardpicks.analysis.logging_utils import (
    setup_analysis_logger,
)
from hardpicks.data.fbp.gather_cleaner import (
    ShotLineGatherCleaner,
)
from hardpicks.data.fbp.gather_parser import (
    create_shot_line_gather_dataset,
)
from hardpicks.data.fbp.site_info import get_site_info_array

logger = logging.getLogger(__name__)
setup_analysis_logger()


def formatting_function(x):
    """Formatting function to get large integers with commas."""
    return "{:,}".format(x)


pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

plt.style.use(style_path)

main_sites = [
    "Sudbury",
    "Halfmile",
    # "Matagami",
    "Brunswick",
    "Lalor"
]

# plots can be fiddly. It's faster to have the needed data on hand.
generate_pickles = False

annotations_path = FBP_BAD_GATHERS_DIR.joinpath("bad-gather-ids_combined.yaml")

pickle_path_template = os.path.join(
    str(pickles_directory), "{site_name}_basic_statistics.pkl"
)
image_path_template = os.path.join(
    str(output_directory), "{site_name}_{line_gather_type}_basic_statistics.png"
)
table_path_template = os.path.join(
    str(output_directory), "{line_gather_type}_basic_statistics_table.tex"
)

if __name__ == "__main__":
    if generate_pickles:
        logger.info("Generating the stats pickles...")

        list_rows = []
        site_info_array = get_site_info_array()
        for site_info in tqdm(site_info_array, desc="SITE"):
            site_name = site_info["site_name"]
            if site_name not in main_sites:
                continue
            raw_dataset = create_shot_line_gather_dataset(
                hdf5_path=site_info["processed_hdf5_path"],
                site_name=site_info["site_name"],
                receiver_id_digit_count=site_info["receiver_id_digit_count"],
                first_break_field_name=site_info["first_break_field_name"],
                provide_offset_dists=True,
            )
            # distinguish "valid" from "clean", where clean means the rejected annotations
            # are taken into account
            valid_dataset = ShotLineGatherCleaner(
                dataset=raw_dataset, auto_invalidate_outlier_picks=False,
            )

            clean_dataset = ShotLineGatherCleaner(
                dataset=raw_dataset,
                auto_invalidate_outlier_picks=False,
                rejected_gather_yaml_path=annotations_path,
            )

            number_of_all_gather_ids = len(np.unique(raw_dataset.trace_to_gather_map))

            valid_stats_df = compute_basic_statistics(valid_dataset)
            clean_stats_df = compute_basic_statistics(clean_dataset)

            number_of_valid_gather_ids = len(valid_dataset.valid_gather_ids)
            number_of_valid_and_clean_gather_ids = len(clean_dataset.valid_gather_ids)

            assert (
                len(valid_stats_df) == number_of_valid_gather_ids
            ), "valid numbers don't match"
            assert (
                len(clean_stats_df) == number_of_valid_and_clean_gather_ids
            ), "clean numbers don't match"
            pickle_path = pickle_path_template.format(site_name=site_name)

            statistics_dict = dict(
                number_of_all_gather_ids=number_of_all_gather_ids,
                number_of_valid_gather_ids=number_of_valid_gather_ids,
                number_of_valid_and_clean_gather_ids=number_of_valid_and_clean_gather_ids,
                valid_stats_df=valid_stats_df,
                clean_stats_df=clean_stats_df,
            )

            with open(pickle_path, "wb") as f:
                pickle.dump(statistics_dict, f)

    logger.info("Generating plots...")

    rows_dict = dict(Valid=[], Useful=[])

    for site_name in tqdm(main_sites, desc="SITE"):

        pickle_path = pickle_path_template.format(site_name=site_name)
        with open(pickle_path, "rb") as f:
            statistics_dict = pickle.load(f)

        valid_stats_df = statistics_dict["valid_stats_df"]
        clean_stats_df = statistics_dict["clean_stats_df"]
        number_of_all_gather_ids = statistics_dict["number_of_all_gather_ids"]
        number_of_valid_gather_ids = statistics_dict["number_of_valid_gather_ids"]
        number_of_valid_and_clean_gather_ids = statistics_dict[
            "number_of_valid_and_clean_gather_ids"
        ]

        for line_gather_type, stats_df in zip(
            ["Valid", "Useful"], [valid_stats_df, clean_stats_df]
        ):
            image_path = image_path_template.format(
                site_name=site_name, line_gather_type=line_gather_type.lower()
            )
            fig = plot_basic_stats(stats_df, line_gather_type=line_gather_type)
            fig.suptitle(
                f"{site_name} Site: Basic Site Statistics for {line_gather_type} Line Gathers"
            )
            fig.savefig(image_path)
            plt.close(fig)

        for key, stats_df in zip(["Valid", "Useful"], [valid_stats_df, clean_stats_df]):
            t1 = f"Total Count Over All {key} Line Gathers"
            t2 = f"Average Count Per {key} Line Gather"
            row = {
                ("", "Site Name"): site_name,
                (t1, "Shots"): len(np.unique(stats_df["shot peg"].values)),
                (t1, "Receiver Lines"): len(np.unique(stats_df["line number"].values)),
                (t1, "Traces"): stats_df["gather size"].sum(),
                (t2, "Invalid Picks"): np.round(stats_df["bad picks"].mean(), 1),
                (t2, "Dead Traces"): np.round(stats_df["dead traces"].mean(), 1),
                (t2, "Traces"): np.round(stats_df["gather size"].mean(), 1),
            }
            rows_dict[key].append(row)

    valid_df = pd.DataFrame(rows_dict["Valid"])
    valid_df.columns = pd.MultiIndex.from_tuples(valid_df.columns)
    useful_df = pd.DataFrame(rows_dict["Useful"])
    useful_df.columns = pd.MultiIndex.from_tuples(useful_df.columns)

    for line_gather_type, df in zip(["Valid", "Useful"], [valid_df, useful_df]):

        trace_column = (
            f"Total Count Over All {line_gather_type} Line Gathers",
            "Traces",
        )
        shot_column = (f"Total Count Over All {line_gather_type} Line Gathers", "Shots")

        formatters = {
            shot_column: formatting_function,
            trace_column: formatting_function,
        }

        latex_output = df.to_latex(
            index=False, multicolumn=True, multicolumn_format="c", formatters=formatters
        )
        # hack the latex output to use tabularx a prettify the table.
        latex_output = (
            latex_output.replace("tabular", "tabularx")
            .replace("{lrrrrrr}", r"{\textwidth}{c *{6}{Y}}")
            .replace(r"Gather} \\", r"Gather} \\\cmidrule(lr){2-4} \cmidrule(l){5-7}")
        )

        output_path = table_path_template.format(
            line_gather_type=line_gather_type.lower()
        )
        with open(output_path, "w") as f:
            f.write(latex_output)
