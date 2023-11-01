import logging
import pickle
import sys
from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from hardpicks import ANALYSIS_RESULTS_DIR
from hardpicks.analysis.fbp.publication.path_constants import style_path,\
    output_directory, pickles_directory
from hardpicks.analysis.fbp.utils import predict_on_test_set, \
    get_fold_config_and_ckpt_paths, rename_site_rms_dataframe_columns

from hardpicks.metrics.fbp.evaluator import FBPEvaluator

plt.style.use(style_path)

logger = logging.getLogger(__name__)

shot_to_rec_offset_norm_const = 3000
site_rms_dataframe_path = ANALYSIS_RESULTS_DIR.joinpath("noise_stats_sites_dataframe.pkl")


def get_test_site_name_for_label(test_site_name):
    """Use correct names in captions."""
    return test_site_name


def plot_hitrate_bins(fold_dataframes, site_rms_dataframe, test_site_dict) -> plt.figure:
    """Plots the hitrate at different noise (RMS amplitude ratios) bins."""
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 4.45))

    axes_list = [*axes[0], *axes[1], *axes[2]]
    for fold_idx, (fold_name, fold_dataframe) in enumerate(fold_dataframes.items()):
        logger.info(f"Creating plot for {fold_name}")
        test_site_name = test_site_dict[fold_name]
        inters_dataframe = deepcopy(pd.merge(
            fold_dataframe,
            site_rms_dataframe[site_rms_dataframe["origin"] == test_site_name],
            how="inner",
            on=["ShotId", "ReceiverId"],
        )[["Errors", "rmsa_ratio"]])
        ax = axes_list[fold_idx]
        inters_dataframe["HR@1px"] = inters_dataframe["Errors"].abs() < 1

        null_error_mask = inters_dataframe["Errors"].isnull()
        if null_error_mask.sum() != 0:
            inters_dataframe["HR@1px"][null_error_mask] = np.nan
        clipped_ratios = inters_dataframe["rmsa_ratio"].clip(upper=1.0)
        ratio_idxs, ratio_bins = pd.cut(clipped_ratios, bins=30, labels=False, retbins=True)
        inters_dataframe["rmsa_bin"] = ratio_idxs
        hitrate_means = inters_dataframe.groupby(["rmsa_bin"]).mean()["HR@1px"]
        ax.plot(
            ratio_bins[1:],
            hitrate_means.values,
            "-",
            marker='+',
            color='red',
            mec='red'
        )
        fold_letter = fold_name[-1].upper()

        ax.set_title(f"Fold {fold_letter} ({get_test_site_name_for_label(test_site_name)})")

        if fold_idx in {6, 7, 8}:
            ax.set_xlabel("Before/After RMS Ratio")
        if fold_idx in {0, 3, 6}:
            ax.set_ylabel("HR@1px")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(False)
    fig.tight_layout()
    return fig


def plot_offset_bins(fold_dataframes, test_site_dict) -> plt.figure:
    """Plots the hitrate at different offset bins."""
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 4.45))
    axes_list = [*axes[0], *axes[1], *axes[2]]
    for fold_idx, (fold_name, fold_dataframe) in enumerate(fold_dataframes.items()):
        logger.info(f"Creating plot for {fold_name}")
        test_site_name = test_site_dict[fold_name]
        ax = axes_list[fold_idx]
        # make a copy so we don't change the input, an undesirable side effect.
        df = deepcopy(fold_dataframe)
        df["HR@1px"] = fold_dataframe["Errors"].abs() < 1

        null_error_mask = fold_dataframe["Errors"].isnull()
        if null_error_mask.sum() != 0:
            df.loc[null_error_mask, "HR@1px"] = np.nan
        offset_idxs, offset_bins = pd.cut(df["Offset"], bins=30, labels=False, retbins=True)
        df["offset_bin"] = offset_idxs
        hitrate_means = df[["HR@1px", "offset_bin"]].groupby("offset_bin").mean(numeric_only=False)["HR@1px"]
        ax.plot(
            offset_bins[1:] * shot_to_rec_offset_norm_const,
            hitrate_means.values,
            "-",
            marker='+',
            color='blue',
            mec='blue'
        )
        fold_letter = fold_name[-1].upper()
        ax.set_title(f"Fold {fold_letter} ({get_test_site_name_for_label(test_site_name)})")
        if fold_idx in {6, 7, 8}:
            ax.set_xlabel("Offset Distance (m)")
        if fold_idx in {0, 3, 6}:
            ax.set_ylabel("HR@1px")

        xmax = 1.05 * np.max(offset_bins[1:][~hitrate_means.isnull()]) * shot_to_rec_offset_norm_const
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.grid(False)
    fig.tight_layout()
    return fig


# plots can be fiddly. It is more convenient to generate the data once and fight with plots after.
fold_dataframes_pickle_path = pickles_directory.joinpath("test_dataframes_dictionary.pkl")
output_ratio_plot_path = output_directory.joinpath("rmsaratio_hitrate_plots.pdf")
output_offset_plot_path = output_directory.joinpath("offset_hitrate_plots.pdf")

# Explicit paths to best models on AWS
test_site_dict = {"foldA": "Matagami",
                  "foldB": "Halfmile",
                  "foldC": "Sudbury",
                  "foldD": "Brunswick",
                  "foldE": "Lalor",
                  "foldH": "Sudbury",
                  "foldI": "Brunswick",
                  "foldJ": "Halfmile",
                  "foldK": "Lalor"}

base_dir_dict = {
    "foldA": "/mnt/efs/experiments/orion/foldA/mlruns/10/35f078ab846c4931a96e3262876398b5/",
    "foldB": "/mnt/efs/experiments/orion/foldB/mlruns/5/9d38db56316c4b4989fddbb7369c8ca1/",
    "foldC": "/mnt/efs/experiments/orion/foldC/mlruns/6/512602b25e7241639eb77402a8fbf922/",
    "foldD": "/mnt/efs/experiments/orion/foldD/mlruns/4/98cded47a0844c319014630e6bd6cf27/",
    "foldE": "/mnt/efs/experiments/orion/foldE/mlruns/6/83356ad80aef49c0bdb0e6dab42d6826/",
    "foldH": "/mnt/efs/experiments/orion/foldH/mlruns/2/8c1921e8064d4d0da5b4959635cffe5c/",
    "foldI": "/mnt/efs/experiments/orion/foldI/mlruns/2/a1c83805bb0a48ce9c31f12d7a8bacee/",
    "foldJ": "/mnt/efs/experiments/orion/foldJ/mlruns/1/0b59afe9340d48a0ac1ebf14f065c145/",
    "foldK": "/mnt/efs/experiments/orion/foldK/mlruns/1/7ca9809ad1a44ec98816c5029e9908f1/"}


generate_pickles = False

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if generate_pickles:
        fold_dataframes = {}
        for fold, test_site_name in test_site_dict.items():
            logger.info(f"Working on fold {fold}")
            fold_root_path = base_dir_dict[fold]

            config_file_path, model_checkpoint_path = get_fold_config_and_ckpt_paths(fold_root_path)

            dataframe_output_path = predict_on_test_set(fold_root_path,
                                                        config_file_path,
                                                        model_checkpoint_path,
                                                        test_site_name)

            evaluator = FBPEvaluator.load(dataframe_output_path)
            print(f"{fold} eval results = {evaluator.summarize()}")
            fold_dataframes[fold] = evaluator._dataframe

        with open(fold_dataframes_pickle_path, 'wb') as fd:
            pickle.dump(fold_dataframes, fd)

    with open(fold_dataframes_pickle_path, 'rb') as fd:
        fold_dataframes = pickle.load(fd)
    # It is assumed that the noise dataframe is available. This is generated from the script
    #   <REPO_ROOT>/hardpicks/analysis/fbp/analyse_site_rms.py
    site_rms_dataframe = rename_site_rms_dataframe_columns(pd.read_pickle(site_rms_dataframe_path))

    fig = plot_hitrate_bins(fold_dataframes, site_rms_dataframe, test_site_dict)
    fig.savefig(output_ratio_plot_path)
    plt.close(fig)
    fig = plot_offset_bins(fold_dataframes, test_site_dict)
    fig.savefig(output_offset_plot_path)
    plt.close(fig)

    print("all done")
