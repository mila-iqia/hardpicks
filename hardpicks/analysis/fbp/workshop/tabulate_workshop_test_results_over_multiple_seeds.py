"""Test Results.

This script produces latex tables for the test results over 10 seeds per fold. The tables contain "mean +/- std"
which make the columns wider. Thus they are split into two: one for the hit rates and one for the regression
metrics.
"""
import logging
from copy import copy

import pandas as pd

from hardpicks.analysis.fbp.mlflow_utils import get_test_results
from hardpicks.analysis.fbp.workshop.path_constants import (
    pickles_directory,
    output_directory,
)
from hardpicks.analysis.logging_utils import (
    setup_analysis_logger,
)

logger = logging.getLogger(__name__)
setup_analysis_logger()

list_metric_names = [
    "test/HitRate1px",
    "test/HitRate3px",
    "test/HitRate5px",
    "test/HitRate7px",
    "test/HitRate9px",
    "test/RootMeanSquaredError",
    "test/MeanAbsoluteError",
    "test/MeanBiasError",
]

list_folds = ["foldH", "foldI", "foldJ", "foldK"]

generate_pickle = True

pickle_path = str(pickles_directory.joinpath("multiple_seeds_test_results.pkl"))

hit_rate_table_output_path = output_directory.joinpath(
    "multiple_seeds_test_results_hit_rate.tex"
)
regression_table_output_path = output_directory.joinpath(
    "multiple_seeds_test_results_regression.tex"
)

test_site_dict = dict(H="Sudbury", I="Brunswick", J="Halfmile", K="Lalor",)

mlflow_tracking_uri_template = "/Users/bruno/monitoring/test_evaluation/{fold}/mlruns"
experiment_name_template = "test_evaluation_{fold}"

metrics_renaming_dictionary = {
    "Fold": ("DEL1", "Fold"),
    "Site": ("DEL2", "Site"),
    "test/HitRate1px": ("HR@", "1px"),
    "test/HitRate3px": ("HR@", "3px"),
    "test/HitRate5px": ("HR@", "5px"),
    "test/HitRate7px": ("HR@", "7px"),
    "test/HitRate9px": ("HR@", "9px"),
    "test/RootMeanSquaredError": ("DEL4", "RMSE"),
    "test/MeanAbsoluteError": ("DEL5", "MAE"),
    "test/MeanBiasError": ("DEL6", "MBE"),
}

postprocessing_substitutions_map = dict()
for i in range(1, 7):
    postprocessing_substitutions_map[f"DEL{i}"] = "    "

if __name__ == "__main__":
    if generate_pickle:
        # Tables are fiddly. Pickle results first.
        results_df = get_test_results(
            mlflow_tracking_uri_template,
            experiment_name_template,
            list_metric_names,
            list_folds,
        )

        results_df.to_pickle(pickle_path)

    results_df = pd.read_pickle(pickle_path)

    groups = results_df.groupby(by="Fold")

    for fold, group_df in groups:
        if len(group_df) != 10:
            logging.warning(f"{fold} does not have 10 runs!")

    mean_df = groups.mean().reset_index()
    std_df = groups.std().reset_index()

    for col in mean_df.columns:
        if "HitRate" in col:
            mean_df[col] = 100 * mean_df[col]
            std_df[col] = 100 * std_df[col]

    mean_df = mean_df.round(decimals=1)
    std_df = std_df.round(decimals=1)

    combined_df = copy(mean_df[["Fold"]])
    for col_name in list_metric_names:
        combined_df[col_name] = (
            mean_df[col_name].map(str) + " SEPARATOR " + std_df[col_name].map(str)
        )

    combined_df["Site"] = combined_df["Fold"].apply(lambda s: test_site_dict[s])

    desired_columns = list(metrics_renaming_dictionary.values())
    combined_df = combined_df.rename(columns=metrics_renaming_dictionary)[
        desired_columns
    ].reset_index(drop=True)
    combined_df.columns = pd.MultiIndex.from_tuples(combined_df.columns)

    columns_hit_rate = pd.MultiIndex.from_tuples(
        [
            ("DEL1", "Fold"),
            ("DEL2", "Site"),
            ("HR@", "1px"),
            ("HR@", "3px"),
            ("HR@", "5px"),
            ("HR@", "7px"),
            ("HR@", "9px"),
        ]
    )
    columns_regression = pd.MultiIndex.from_tuples(
        [
            ("DEL1", "Fold"),
            ("DEL2", "Site"),
            ("DEL4", "RMSE"),
            ("DEL5", "MAE"),
            ("DEL6", "MBE"),
        ]
    )

    for columns, table_output_path in zip(
        [columns_hit_rate, columns_regression],
        [hit_rate_table_output_path, regression_table_output_path],
    ):
        latex_output = combined_df[columns].to_latex(
            index=False, multicolumn=True, multicolumn_format="c"
        )
        for key, value in postprocessing_substitutions_map.items():
            latex_output = latex_output.replace(key, value)
        latex_output = latex_output.replace("SEPARATOR", r"$\pm$")

        with open(table_output_path, "w") as f:
            f.write(latex_output)
