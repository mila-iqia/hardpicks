"""Plot error distributions and confidence scatter plots.

This script plots, for each fold and each dataset, a figure with the error histogram as the top pane and the
probability vs. error as the bottom pane, for the geophysics paper. It also splits the errors according to
"in the main image" or "in the padding". The script also outputs to console some aggregated error metrics
for errors in or not in the padding.


NOTE ADDED IN AUGUST 2023: Running this code requires downgrading pandas to <2.0.0, as newer
versions of the pandas library are not compatible with the data pickles on disk.
"""
import pickle
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from hardpicks.analysis.fbp.all_folds_results_paths import (
    get_all_folds_results_pickle_path_and_info,
)
from hardpicks.analysis.fbp.publication.path_constants import (
    style_path,
    output_directory,
)
from hardpicks.plotting import PLEASANT_FIG_SIZE

plt.style.use(style_path)


predictions_base_dir = Path(
    "/Users/bruno/monitoring/FBP/supplementary_experiments/supplement-predict/"
)


if __name__ == "__main__":

    list_results_and_site_info = get_all_folds_results_pickle_path_and_info(
        predictions_base_dir
    )

    output_file = open(output_directory / "PADDING_ERRORS_COUNTS.txt", "w")

    for results_and_site_info in tqdm(list_results_and_site_info, "FOLD"):
        site_name = results_and_site_info.site_name
        fold = results_and_site_info.fold
        dataset = results_and_site_info.dataset
        number_of_samples = results_and_site_info.number_of_samples
        path_to_evaluator_pickle = results_and_site_info.path_to_data_pickle

        with open(path_to_evaluator_pickle, "rb") as fd:
            attribs = pickle.load(fd)

        df = attribs["dataframe"]
        annotated_df = df[~df["Errors"].isna()]
        error_series = annotated_df["Errors"]
        probability_series = 100.0 * annotated_df["Probabilities"]

        xmin = np.floor(error_series.min() / 100) * 100
        xmax = np.ceil(error_series.max() / 100) * 100

        print(f"Fold {fold}, dataset {dataset}, site {site_name}", file=output_file)
        print(f"    - There are {len(df)} total traces.", file=output_file)
        print(f"    - There are {len(annotated_df)} traces.", file=output_file)

        valid_prediction_mask = annotated_df["Predictions"] < number_of_samples
        print(
            f"    - There are {valid_prediction_mask.sum()} predictions NOT IN THE PADDING on annotated traces.",
            file=output_file,
        )

        all_errors = annotated_df["Errors"].values
        no_padding_errors = annotated_df[valid_prediction_mask]["Errors"].values

        for key, errors in zip(
            ["ALL TRACES", "NO PADDING ERROR TRACES"], [all_errors, no_padding_errors]
        ):

            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(np.abs(errors))
            mbe = np.mean(errors)
            tc = len(errors) / len(annotated_df)

            print(
                f"    - {key}: RMSE = {rmse:2.1f} MAE = {mae: 2.1f}  MBE = {mbe:2.1f} TRACE COVERAGE = {tc:5.3f}",
                file=output_file,
            )

        pleasant_half_height_size = (PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1] / 2)
        fig1 = plt.figure(figsize=pleasant_half_height_size)
        fig2 = plt.figure(figsize=pleasant_half_height_size)

        ax1 = fig1.add_subplot(111)

        ax1.hist(
            [
                error_series[valid_prediction_mask].values,
                error_series[~valid_prediction_mask].values,
            ],
            color=["blue", "red"],
            alpha=0.5,
            bins=50,
            label=["Predictions not in Padding", "Predictions in Padding"],
            stacked=True,
        )
        ax1.set_yscale("log")
        ax1.set_xlim(xmin=xmin, xmax=xmax)
        ax1.legend(loc=0)
        ax1.set_xlabel("Error (pixels)")
        ax1.set_ylabel("Count")

        ax2 = fig2.add_subplot(111)

        ax2.scatter(
            error_series[valid_prediction_mask].values,
            probability_series[valid_prediction_mask].values,
            edgecolor="black",
            facecolor="blue",
            alpha=0.5,
            label="Errors not in Padding",
        )
        ax2.scatter(
            error_series[~valid_prediction_mask].values,
            probability_series[~valid_prediction_mask].values,
            edgecolor="black",
            facecolor="red",
            alpha=0.5,
            label="Errors in Padding",
        )

        ax2.set_xlabel("Error (pixels)")
        ax2.set_ylabel("Probability (%)")
        ax2.set_ylim([0, 100])
        ax2.set_xlim([xmin, xmax])
        # Don't put a legend, it overlaps with the plotted data.
        # ax2.legend(loc=0)

        for fig in [fig1, fig2]:
            fig.tight_layout()

        image_path1 = output_directory.joinpath(
            f"{fold}_{dataset}_{site_name}_error_distribution.png"
        )
        image_path2 = output_directory.joinpath(
            f"{fold}_{dataset}_{site_name}_probability_scatter.png"
        )
        fig1.savefig(image_path1)
        fig2.savefig(image_path2)

        plt.close(fig1)
        plt.close(fig2)

    output_file.close()
