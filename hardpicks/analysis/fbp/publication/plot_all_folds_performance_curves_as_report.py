"""Plot for all folds.

This script plots a figure with the error histogram as the top pane and the probability vs. error as the bottom
pane, for each fold, for the validation and the test dataset. These plots are all appended to a single
pdf report.
"""
import glob
import pickle
from datetime import date
from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd
from tqdm import tqdm

from hardpicks import ANALYSIS_RESULTS_DIR, ROOT_DIR
from hardpicks.analysis.fbp.publication.path_constants import (
    style_path,
)
from hardpicks.metrics.fbp.evaluator import FBPEvaluator
from hardpicks.plotting import PLEASANT_FIG_SIZE
from hardpicks.utils.hash_utils import get_git_hash
import numpy as np

plt.style.use(style_path)


path_to_this_file = str(Path(__file__).relative_to(ROOT_DIR))

list_folds = [
    "foldA",
    "foldB",
    "foldC",
    "foldD",
    "foldE",
    "foldH",
    "foldI",
    "foldJ",
    "foldK",
]

# This folder should contain all the fold predictions in subfolders of the form foldA, foldB, etc...
all_folds_predictions_base_dir = Path(
    "/Users/bruno/monitoring/FBP/supplementary_experiments/supplement-predict/"
)

if __name__ == "__main__":

    today = date.today()
    git_revision = get_git_hash()
    front_page_series = pd.Series(
        [today, git_revision, path_to_this_file],
        index=["Date", "git revision", "script"],
    )

    list_image_paths = []
    for fold in tqdm(list_folds, "FOLD"):
        for dataset in ["valid", "test"]:
            data_dump_path = glob.glob(
                str(all_folds_predictions_base_dir / fold / f"*_{dataset}.pkl")
            )[0]

            with open(data_dump_path, "rb") as fd:
                attribs = pickle.load(fd)

            evaluator = FBPEvaluator.load(data_dump_path)
            site_name = list(evaluator.origin_id_map.keys())[0]

            df = attribs["dataframe"]

            label_df = df[~df["Errors"].isna()]
            max_p = label_df["Probabilities"].max()
            list_thresholds = np.linspace(0.0, max_p, 100)
            list_thresholds = np.append(list_thresholds, 1.0)

            total_number_of_labels = len(label_df)
            mask_no_errors = label_df["Errors"] == 0.0
            list_trace_coverage = []
            list_precision = []
            list_hit_rate = []
            for threshold in list_thresholds:
                mask_threshold = label_df["Probabilities"] >= threshold
                number_of_predictions = mask_threshold.sum()

                trace_coverage = number_of_predictions / total_number_of_labels
                list_trace_coverage.append(trace_coverage)

                number_of_correct_predictions = (mask_threshold * mask_no_errors).sum()
                precision = number_of_correct_predictions / number_of_predictions
                list_precision.append(precision)

                hit_rate = number_of_correct_predictions / total_number_of_labels
                list_hit_rate.append(hit_rate)

            list_trace_coverage = 100 * np.array(list_trace_coverage)
            list_precision = 100 * np.array(list_precision)
            list_hit_rate = 100 * np.array(list_hit_rate)

            fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
            fig.suptitle(f"{fold}, {dataset} dataset, site {site_name}")
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(323)
            ax3 = fig.add_subplot(325)
            ax4 = fig.add_subplot(122)

            ax1.plot(100 * list_thresholds, list_trace_coverage, "b-")
            ax1.set_xlabel("Confidence Threshold (%)")
            ax1.set_ylabel("Trace Coverage (%)")
            ymin = np.max([np.nanmin(list_trace_coverage) - 1., -1.])
            ymax = np.min([np.nanmax(list_trace_coverage) + 1., 101.])
            ax1.set_ylim(ymin, ymax)

            ax2.plot(100 * list_thresholds, list_precision, "b-")
            ax2.set_xlabel("Confidence Threshold (%)")
            ax2.set_ylabel("Precision (%)")
            ymin = np.max([np.nanmin(list_precision) - 1., -1.])
            ymax = np.min([np.nanmax(list_precision) + 1., 101.])
            ax2.set_ylim(ymin, ymax)

            ax3.plot(100 * list_thresholds, list_hit_rate, "b-")
            ax3.set_xlabel("Confidence Threshold (%)")
            ax3.set_ylabel("Hit Rate (%)")
            ymin = np.max([np.nanmin(list_hit_rate) - 1., -1.])
            ymax = np.min([np.nanmax(list_hit_rate) + 1., 101.])
            ax3.set_ylim(ymin, ymax)

            ax4.plot(list_trace_coverage, list_precision, "bo-")
            ax4.set_xlabel("Trace Coverage (%)")
            ax4.set_ylabel("Precision (%)")
            ax4.set_ylim(-1, 101)

            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlim(-1, 101)

            fig.tight_layout()

            image_path = (
                ANALYSIS_RESULTS_DIR / f"{fold}_{dataset}_{site_name}_performance.png"
            )
            fig.savefig(image_path)
            plt.close(fig)
            list_image_paths.append(image_path)

    # pdf_output_path = ANALYSIS_RESULTS_DIR.joinpath("all_folds_performance_curves.pdf")
    # write_report_and_cleanup(pdf_output_path, list_image_paths, front_page_series)
