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
            error_series = df["Errors"]
            xmin = error_series.min() - 10
            xmax = error_series.max() + 10

            fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
            fig.suptitle(f"{fold}, {dataset} dataset, site {site_name}")
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            ax1.hist(error_series, color="blue", bins=50)
            ax1.set_yscale("log")
            ax1.set_xlabel("Error (pixels)")
            ax1.set_ylabel("Count")

            df = attribs["dataframe"][["Errors", "Probabilities"]].dropna()
            df["Probabilities"] = 100 * df["Probabilities"]

            df.plot(x="Errors", y="Probabilities", kind="scatter", color="blue", ax=ax2)
            ax2.set_xlabel("Error (pixels)")
            ax2.set_ylabel("Probability (%)")
            ax2.set_ylim([0, 100])

            for ax in [ax1, ax2]:
                ax.set_xlim([xmin, xmax])

            fig.tight_layout()

            image_path = (
                ANALYSIS_RESULTS_DIR
                / f"{fold}_{dataset}_{site_name}_error_distribution_and_probability_scatter.png"
            )
            fig.savefig(image_path)
            plt.close(fig)
            list_image_paths.append(image_path)

    # pdf_output_path = ANALYSIS_RESULTS_DIR.joinpath(
    #     "all_folds_error_distribution_and_probability_scatter.pdf"
    # )
    # write_report_and_cleanup(pdf_output_path, list_image_paths, front_page_series)
