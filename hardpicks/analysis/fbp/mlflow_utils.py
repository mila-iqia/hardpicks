import logging
import os

import pandas as pd

from hardpicks.analysis.mlflow_and_orion_analysis.mlflow_utils import \
    get_client_and_experiment_id, get_metric_series


def get_test_results(
    mlflow_tracking_uri_template,
    experiment_name_template,
    list_metric_names,
    list_folds,
):
    """Extract test results from mlflow.

    Args:
        mlflow_tracking_uri_template: string indicating the path of the mlfow data, of the form
                                            "/path/to/folder/{fold}/mlruns"
                                      where {fold} can be substituted by the correct fold name within
                                      this function.
        experiment_name_template: string with the name of the experiment, of the form
                                        "test_evaluation_{fold}"
                                  where again {fold} can be substituted by the correct fold name
        list_metric_names: list of metrics to be extracted.
        list_folds: list of fold names, like ["foldA", "foldB", etc...].

    Returns:
        df: a pandas dataframe with all the test results.
    """
    list_rows = []
    for fold in list_folds:
        logging.info(f"  - doing {fold}...")
        tracking_uri = mlflow_tracking_uri_template.format(fold=fold)
        experiment_name = experiment_name_template.format(fold=fold)

        client, experiment_id = get_client_and_experiment_id(
            tracking_uri, experiment_name
        )

        run_folder_path = os.path.join(tracking_uri, experiment_id)
        list_run_ids = [
            run_id for run_id in os.listdir(run_folder_path) if run_id != "meta.yaml"
        ]

        for run_id in list_run_ids:
            logging.info(f"  - doing {run_id}...")

            try:
                run = client.get_run(run_id)
                row = {"Fold": fold.replace("fold", ""), "run_id": run_id}
                for metric_name in list_metric_names:
                    metric_series = get_metric_series(client, run.info, metric_name)
                    assert len(metric_series) == 1, "Something is wrong"
                    row[metric_name] = metric_series.values[0]
                list_rows.append(row)
            except Exception:
                logging.warning(f"No test result found for {fold} - {run_id}")
                continue

    df = pd.DataFrame(list_rows)
    return df
