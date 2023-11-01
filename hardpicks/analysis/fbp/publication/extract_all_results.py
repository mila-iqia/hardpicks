import logging
import pandas as pd
import pickle

from hardpicks.analysis.fbp.publication.path_constants import pickles_directory
from hardpicks.analysis.mlflow_and_orion_analysis.results_utils import get_completed_results
from hardpicks.analysis.logging_utils import (
    setup_analysis_logger,
)

logger = logging.getLogger(__name__)
setup_analysis_logger()

pickle_path = str(pickles_directory.joinpath("results.pkl"))

list_folds = ["foldA", "foldB", "foldC", "foldD", "foldE", "foldH", "foldI", "foldJ", "foldK"]

experiment_name = "annotation_fix"
extra_experiment_name_foldC = "annotation_fix_v2"

early_stopping_metric_name = "valid/HitRate1px"
list_metric_names = [
    "valid/HitRate1px",
    "valid/HitRate3px",
    "valid/HitRate5px",
    "valid/HitRate7px",
    "valid/HitRate9px",
    "epoch",
    "valid/RootMeanSquaredError",
    "valid/GatherCoverage",
    "valid/MeanAbsoluteError",
    "valid/MeanBiasError",
]

results_dict = dict()
if __name__ == "__main__":

    for fold in list_folds:
        logger.info(f"Doing fold {fold}")
        orion_database_path = f"/Users/bruno/monitoring/FBP/orion/{fold}/orion_db.pkl"
        mlflow_tracking_uri = f"/Users/bruno/monitoring/FBP/orion/{fold}/mlruns"

        completed_results = get_completed_results(
            orion_database_path,
            experiment_name,
            mlflow_tracking_uri,
            early_stopping_metric_name,
            list_metric_names,
        )
        if fold == "foldC":
            completed_results2 = get_completed_results(
                orion_database_path,
                extra_experiment_name_foldC,
                mlflow_tracking_uri,
                early_stopping_metric_name,
                list_metric_names,
            )
            completed_results = pd.concat(
                [completed_results, completed_results2]
            ).reset_index(drop=True)

        results_dict[fold] = completed_results.sort_values(
            by=early_stopping_metric_name, ascending=False
        ).head(50)

    logger.info("Dumping results to pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(results_dict, f)
