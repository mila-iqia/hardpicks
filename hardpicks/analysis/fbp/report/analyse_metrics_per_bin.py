import logging
import os
import sys

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from hardpicks import ANALYSIS_RESULTS_DIR, FBP_DATA_DIR
from hardpicks.analysis.fbp.report.path_constants import output_directory
from hardpicks.analysis.fbp.utils import predict_on_test_set, \
    rename_site_rms_dataframe_columns, get_fold_config_and_ckpt_paths
from hardpicks.metrics.fbp.evaluator import FBPEvaluator

logger = logging.getLogger(__name__)

output_ratio_plot_path = output_directory.joinpath("rmsaratio_hitrate_plots.pdf")
output_offset_plot_path = output_directory.joinpath("offset_hitrate_plots.pdf")
best_ckpt_export_root_path = os.path.join(FBP_DATA_DIR, "best_ckpt_export")
target_folds_and_test_sets = [
    ("fold_a", "Matagami"),
    ("fold_b", "Halfmile"),
    ("fold_c", "Sudbury"),
    ("fold_d", "Brunswick"),
    ("fold_e", "Lalor"),
    ("fold_h", "Sudbury"),
    ("fold_i", "Brunswick"),
    ("fold_j", "Halfmile"),
]
shot_to_rec_offset_norm_const = 3000
site_rms_dataframe_path = ANALYSIS_RESULTS_DIR.joinpath("noise_stats_sites_dataframe.pkl")


def plot_hitrate_bins(fold_dataframes, site_rms_dataframe) -> plt.figure:
    """Plots the hitrate at different noise (RMS amplitude ratios) bins."""
    fig, axes = plt.subplots(2, 4, figsize=(9, 5.5))
    axes_list = [*axes[0], *axes[1]]
    for fold_idx, (fold_name, fold_dataframe) in enumerate(fold_dataframes.items()):
        test_site_name = target_folds_and_test_sets[fold_idx][1]
        inters_dataframe = pd.merge(
            fold_dataframe,
            site_rms_dataframe[site_rms_dataframe["origin"] == test_site_name],
            how="inner",
            on=["ShotId", "ReceiverId"],
        )[["Errors", "rmsa_ratio"]]
        ax = axes_list[fold_idx]
        inters_dataframe["HR@1px"] = inters_dataframe["Errors"].abs() < 1
        inters_dataframe["HR@1px"][inters_dataframe["Errors"].isnull()] = np.nan
        clipped_ratios = inters_dataframe["rmsa_ratio"].clip(upper=1.0)
        ratio_idxs, ratio_bins = pd.cut(clipped_ratios, bins=30, labels=False, retbins=True)
        inters_dataframe["rmsa_bin"] = ratio_idxs
        hitrate_means = inters_dataframe.groupby(["rmsa_bin"]).mean()["HR@1px"]
        ax.plot(
            ratio_bins[1:],
            hitrate_means.values,
            "r+-",
        )
        fold_letter = fold_name[-1].upper()
        ax.set_title(f"Fold {fold_letter} ({test_site_name})")
        ax.set_xlabel("Before/After RMS Ratio")
        ax.set_ylabel("HR@1px")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(False)
    fig.tight_layout()
    return fig


def plot_offset_bins(fold_dataframes) -> plt.figure:
    """Plots the hitrate at different offset bins."""
    fig, axes = plt.subplots(2, 4, figsize=(9, 5.5))
    axes_list = [*axes[0], *axes[1]]
    for fold_idx, (fold_name, fold_dataframe) in enumerate(fold_dataframes.items()):
        test_site_name = target_folds_and_test_sets[fold_idx][1]
        ax = axes_list[fold_idx]
        fold_dataframe["HR@1px"] = fold_dataframe["Errors"].abs() < 1
        fold_dataframe["HR@1px"][fold_dataframe["Errors"].isnull()] = np.nan
        offset_idxs, offset_bins = pd.cut(fold_dataframe["Offset"], bins=30, labels=False, retbins=True)
        fold_dataframe["offset_bin"] = offset_idxs
        hitrate_means = fold_dataframe.groupby(["offset_bin"]).mean()["HR@1px"]
        ax.plot(
            offset_bins[1:] * shot_to_rec_offset_norm_const,
            hitrate_means.values,
            "b+-",
        )
        fold_letter = fold_name[-1].upper()
        ax.set_title(f"Fold {fold_letter} ({test_site_name})")
        ax.set_xlabel("Offset Distance (m)")
        ax.set_ylabel("HR@1px")
        ax.set_ylim(0, 1)
        ax.grid(False)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    fold_dataframes = {}
    for target_fold, test_site_name in target_folds_and_test_sets:
        fold_root_path = os.path.join(best_ckpt_export_root_path, target_fold)
        config_file_path, model_checkpoint_path = get_fold_config_and_ckpt_paths(fold_root_path)

        dataframe_output_path = predict_on_test_set(fold_root_path,
                                                    config_file_path,
                                                    model_checkpoint_path,
                                                    test_site_name)

        evaluator = FBPEvaluator.load(dataframe_output_path)
        print(f"{target_fold} eval results = {evaluator.summarize()}")
        fold_dataframes[target_fold] = evaluator._dataframe
    site_rms_dataframe = rename_site_rms_dataframe_columns(pd.read_pickle(site_rms_dataframe_path))
    fig = plot_hitrate_bins(fold_dataframes, site_rms_dataframe)
    fig.savefig(output_ratio_plot_path)
    plt.close(fig)
    fig = plot_offset_bins(fold_dataframes)
    fig.savefig(output_offset_plot_path)
    plt.close(fig)

    print("all done")
