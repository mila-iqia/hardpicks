import logging

import pandas as pd

from hardpicks import ANALYSIS_RESULTS_DIR
from hardpicks.analysis.logging_utils import setup_analysis_logger
from hardpicks.analysis.mlflow_and_orion_analysis.mlflow_utils import get_mlflow_results

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

logger = logging.getLogger(__name__)
setup_analysis_logger()

early_stopping_metric_name = "valid/HitRate1px"
list_metric_names = ['valid/HitRate1px', 'valid/HitRate3px', 'valid/HitRate5px', 'valid/HitRate7px',
                     'valid/HitRate9px', 'epoch', 'valid/RootMeanSquaredError', 'valid/GatherCoverage',
                     'valid/MeanAbsoluteError', 'valid/mIoU', 'valid/MeanBiasError']

experiment_name = "annotation_fix"

if experiment_name == 'annotation_fix_v2':
    list_folds = ["foldC"]
elif experiment_name == 'annotation_fix':
    list_folds = ["foldA", "foldB", "foldC", "foldD", "foldE", "foldH", "foldI", "foldJ", "foldK"]
elif experiment_name == 'data_fix':
    list_folds = ["foldA", "foldB", "foldC", "foldD", "foldE", "foldH", "foldI"]
elif experiment_name == 'large_search':
    list_folds = ["foldA", "foldB", "foldC", "foldD", "foldE"]
elif experiment_name == 'new_brunswick_data':
    list_folds = ["foldA", "foldB", "foldC", "foldE"]
elif experiment_name == 'no_lalor':
    list_folds = ["foldF", "foldG"]
else:
    raise ValueError("experiment is not accounted for.")


summary_results_directory = ANALYSIS_RESULTS_DIR.joinpath("results_summary")
summary_results_directory.mkdir(exist_ok=True)
output_path = summary_results_directory.joinpath(f"best_{experiment_name}_results.csv")

if __name__ == "__main__":
    logger.info(f"Extracting best results for experiment {experiment_name}")
    # list_rows = []
    list_dataframes = []
    for fold in list_folds:
        logger.info(f"doing fold {fold}...")

        tracking_uri = f"/Users/bruno/monitoring/orion/{fold}/mlruns"
        fold_df = get_mlflow_results(tracking_uri, experiment_name, early_stopping_metric_name, list_metric_names)
        fold_df['fold'] = fold
        list_dataframes.append(fold_df)

    df = pd.concat(list_dataframes).reset_index(drop=True)

    fold_groups = df.groupby(by="fold")
    job_count_df = pd.DataFrame(fold_groups["fold"].count()).rename(columns={"fold": "Job Count"})
    summary_df = df.iloc[fold_groups[early_stopping_metric_name].idxmax()].set_index('fold')
    summary_df = pd.merge(job_count_df, summary_df, left_index=True, right_index=True)

    print("Maximum performance per fold:")
    print(summary_df)
    summary_df.to_csv(output_path)
