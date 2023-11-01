"""Describe padding predictions for all folds and all datasets.

This simple script just prints out the unique padding predictions.
Spoiler alert, for Lalor - foldD, they are all at 2047, the end of the padding region.
"""
from pathlib import Path

import numpy as np

from hardpicks.analysis.fbp.all_folds_results_paths import \
    get_all_folds_results_pickle_path_and_info
from hardpicks.metrics.fbp.evaluator import FBPEvaluator

predictions_base_dir = Path("/Users/bruno/monitoring/FBP/supplementary_experiments/supplement-predict/")


if __name__ == "__main__":

    results_and_site_info = get_all_folds_results_pickle_path_and_info(predictions_base_dir)

    for info in results_and_site_info:

        evaluator = FBPEvaluator.load(info.path_to_data_pickle)
        evaluator_df = evaluator._dataframe.reset_index(drop=True)

        annotated_df = evaluator_df[~evaluator_df['Errors'].isna()]

        number_of_samples_with_padding = 2**int(np.ceil(np.log2(info.number_of_samples)))

        predictions_series = annotated_df['Predictions']
        number_of_annotated_traces = len(predictions_series)
        padding_predictions_series = predictions_series[predictions_series >= info.number_of_samples]
        number_of_padding_predictions = len(padding_predictions_series)
        unique_padding_predictions = padding_predictions_series.unique()

        ratio_percentage = 100. * number_of_padding_predictions / number_of_annotated_traces

        print("=================================================================================")
        print(f"    Fold : {info.fold}")
        print(f" Dataset : {info.dataset}")
        print(f"    Site : {info.site_name}")
        print(f" samples : {info.number_of_samples}")
        print(f" samples with padding: {number_of_samples_with_padding}")
        print(f'            number of annotated traces    : {number_of_annotated_traces}')
        print(f'            number of padding predictions : {number_of_padding_predictions}')
        print(f'            fraction of padding predictions : {ratio_percentage:6.5f} %')
        print(f'Unique padding predictions (in px): {unique_padding_predictions}')
