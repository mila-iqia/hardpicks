import logging
import os
from copy import deepcopy

import pandas as pd

from hardpicks.analysis.mlflow_and_orion_analysis.orion_utils import (
    get_orion_experiment_dataframe,
)
from hardpicks.analysis.mlflow_and_orion_analysis.parallel_coordinates_utils import (
    process_orion_dataframe,
)
from hardpicks.analysis.fbp.publication.path_constants import (
    output_directory,
)

output_path_template = os.path.join(
    str(output_directory), "best_validation_configuration_table_{fold}.tex"
)

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


experiment_name = "annotation_fix"
extra_experiment_name_foldC = "annotation_fix_v2"
new_objective_name = "HitRate@1px"

if __name__ == "__main__":

    list_top_df = []
    for fold in list_folds:
        logging.info(f"Doing fold {fold}")
        path_to_db = f"/Users/bruno/monitoring/orion/{fold}/orion_db.pkl"

        orion_df = get_orion_experiment_dataframe(path_to_db, experiment_name)
        df = process_orion_dataframe(orion_df, new_objective_name)

        if fold == "foldC":
            orion_df = get_orion_experiment_dataframe(
                path_to_db, extra_experiment_name_foldC
            )
            df2 = process_orion_dataframe(orion_df, new_objective_name)
            df = (
                pd.concat([df, df2])
                .sort_values(by=new_objective_name, ascending=False)
                .head(50)
            )

        latex_df = df.head(10).reset_index(drop=True)
        latex_df[new_objective_name] = (100 * latex_df[new_objective_name]).round(
            decimals=1
        )

        top_df = deepcopy(latex_df.head(1))
        top_df["Fold"] = fold.replace("fold", "")
        list_top_df.append(top_df)

    all_folds_df = pd.concat(list_top_df).set_index("Fold").reset_index()
    del all_folds_df[new_objective_name]
    output_path = output_path_template.format(fold="all")
    latex_output = (
        all_folds_df.to_latex(index=False).replace("[", "$[").replace("]", "]$")
    )
    with open(output_path, "w") as f:
        f.write(latex_output)
