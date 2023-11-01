import pandas as pd

from hardpicks.analysis.mlflow_and_orion_analysis.mlflow_utils import (
    get_mlflow_results,
)
from hardpicks.analysis.mlflow_and_orion_analysis.orion_utils import (
    get_orion_database_results,
)


def get_completed_results(
    orion_database_path,
    experiment_name,
    mlflow_tracking_uri,
    early_stopping_metric_name,
    list_metric_names,
):
    """Extract the results from orion and mlflow, and combine."""
    orion_df = get_orion_database_results(orion_database_path, experiment_name)
    mlflow_df = get_mlflow_results(
        mlflow_tracking_uri,
        experiment_name,
        early_stopping_metric_name,
        list_metric_names,
    )
    results_df = pd.merge(mlflow_df, orion_df, on="orion_id", how="outer")

    completed_mask = results_df["status"] == "completed"
    completed_df = results_df[completed_mask]
    return completed_df
