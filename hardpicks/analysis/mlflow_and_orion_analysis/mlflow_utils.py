import logging
import os

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def get_client_and_experiment_id(tracking_uri: str, experiment_name: str):
    """Extract mlflow client and experiment id using mlflow's api."""
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = None
    for candidate in client.list_experiments():
        if candidate.name == experiment_name:
            experiment = candidate
            break

    if experiment is None:
        raise Exception(f"Experiment {experiment_name} not found tracking uri {tracking_uri}!.")

    return client, experiment.experiment_id


def get_metric_series(client, run_info, metric_name):
    """Extract pandas series for a metric using mlflow's api."""
    result_dict = dict()
    for metric in client.get_metric_history(run_info.run_id, metric_name):
        result_dict[metric.step] = metric.value
    series = pd.Series(result_dict, name=metric_name)
    series.index.name = 'step'
    return series


def get_mlflow_results(tracking_uri, experiment_name, early_stopping_metric_name, list_metric_names):
    """Extract all results available for an mlflow experiment."""
    logger.info("  Getting client and experiment id...")
    client, experiment_id = get_client_and_experiment_id(tracking_uri, experiment_name)

    run_folder_path = os.path.join(tracking_uri, experiment_id)
    list_run_ids = [run_id for run_id in os.listdir(run_folder_path) if run_id != 'meta.yaml']

    list_rows = []
    for run_id in list_run_ids:
        logger.info(f"  - doing {run_id}...")

        try:
            run = client.get_run(run_id)
            run_name = run.data.tags["mlflow.runName"]

            list_metrics_present_in_run = list(run.data.metrics.keys())
            list_series = []
            for metric_name in list_metric_names:
                if metric_name not in list_metrics_present_in_run:
                    continue
                metric_series = get_metric_series(client, run.info, metric_name)
                list_series.append(metric_series)

            run_df = pd.concat(list_series, axis=1, join='inner')
            best_idx = run_df[early_stopping_metric_name].argmax()
            row = dict(run_df.iloc[best_idx])

            row.update({"run_id": run_id,
                        "orion_id": run_name,
                        })
            list_rows.append(row)
        except Exception:
            logging.error(f"Problem encountered for run id {run_id}")

    mlflow_df = pd.DataFrame(list_rows)
    return mlflow_df
