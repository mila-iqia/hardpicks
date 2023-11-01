import os
import pickle

import pandas as pd

from hardpicks.analysis.fbp.workshop.path_constants import (
    output_directory,
    pickles_directory,
)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


validation_site_dict = dict(H="Brunswick", I="Lalor", J="Sudbury", K="Halfmile",)

# Here we use place holders DEL1,... to force pandas to create correct columns.
# We will remove these placeholders after
metrics_renaming_dictionary = {
    "Fold": ("DEL1", "Fold"),
    "Site": ("DEL2", "Site"),
    "valid/HitRate1px": ("HR@", "1px"),
    "valid/HitRate3px": ("HR@", "3px"),
    "valid/HitRate5px": ("HR@", "5px"),
    "valid/HitRate7px": ("HR@", "7px"),
    "valid/HitRate9px": ("HR@", "9px"),
    "valid/RootMeanSquaredError": ("DEL4", "RMSE"),
    "valid/MeanAbsoluteError": ("DEL5", "MAE"),
    "valid/MeanBiasError": ("DEL6", "MBE"),
}

postprocessing_substitutions_map = dict()
for i in range(1, 7):
    postprocessing_substitutions_map[f"DEL{i}"] = "    "


output_path = os.path.join(str(output_directory), "validation_results_table.tex")

results_pickle_path = str(pickles_directory.joinpath("results.pkl"))
early_stopping_metric_name = "valid/HitRate1px"

if __name__ == "__main__":
    with open(results_pickle_path, "rb") as f:
        results_dict = pickle.load(f)

    list_dataframes = []
    for fold, fold_df in results_dict.items():
        fold_df["Fold"] = fold.replace("fold", "")
        list_dataframes.append(fold_df)

    df = pd.concat(list_dataframes).reset_index(drop=True)

    fold_groups = df.groupby(by="Fold")
    job_count_df = pd.DataFrame(fold_groups["Fold"].count()).rename(
        columns={"Fold": "Job Count"}
    )
    summary_df = df.iloc[fold_groups[early_stopping_metric_name].idxmax()].set_index(
        "Fold"
    )
    summary_df = pd.merge(
        job_count_df, summary_df, left_index=True, right_index=True
    ).reset_index()
    summary_df["Site"] = summary_df["Fold"].apply(lambda s: validation_site_dict[s])

    print("Maximum performance per fold:")
    print(summary_df)

    desired_columns = list(metrics_renaming_dictionary.values())
    output_df = summary_df.rename(columns=metrics_renaming_dictionary)[
        desired_columns
    ].reset_index(drop=True)

    for col in desired_columns:
        if "HR@" in col[0]:
            output_df[col] = 100 * output_df[col]
    output_df = output_df.round(decimals=1)

    latex_output = output_df.to_latex(
        index=False, multicolumn=True, multicolumn_format="c"
    )
    for key, value in postprocessing_substitutions_map.items():
        latex_output = latex_output.replace(key, value)

    with open(output_path, "w") as f:
        f.write(latex_output)
