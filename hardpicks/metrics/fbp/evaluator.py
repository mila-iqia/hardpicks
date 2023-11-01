"""First Break Picking (FBP) metric evaluator module.

This module implements the evaluator for the FBP metrics.
"""

import copy
import logging
import pickle
import typing

import numpy as np
import pandas as pd
import torch

import hardpicks.metrics.base as eval_base
import hardpicks.metrics.fbp.utils as utils
from hardpicks.utils.hp_utils import check_hp
from hardpicks.utils.profiling_utils import profile

logger = logging.getLogger(__name__)

SUPPORTED_SEGMENTATION_METRICS = [
    "GatherCoverage",  # computed across all classes
]

SUPPORTED_REGRESSION_METRICS = [
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "MeanBiasError",
    "HitRate",  # can be defined with various pixel thresholds
]

SUPPORTED_METRICS = [
    *SUPPORTED_SEGMENTATION_METRICS,
    *SUPPORTED_REGRESSION_METRICS,
]

LOWER_IS_BETTER_METRICS = [
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "MeanBiasError",
    "loss",
]

HIGHER_IS_BETTER_METRICS = [
    "HitRate",
    "GatherCoverage",
]


class FBPEvaluator(eval_base.EvaluatorBase):
    """First Break Picking (FBP) evaluator implementation.

    This evaluator is used for FBP-specific minibatches; it will automatically create categories
    based on site names, receiver-shot distances, and acquisition season. It supports both
    segmentation and regression metrics, and will derive first breaks from segmentation maps
    if the latter type of metrics are used.

    All the ingested prediction results will be transformed into evaluation results, and these will
    be stored in a dataframe for easier summarization and categorization of metrics at the end of
    each epoch.

    Attributes:
        origin_id_map: map that translates origin (i.e. site names taken from HDF5 file names) into
            ids that will help minimize the size of the stored entries in the internal dataframe.
        seen_batch_idxs: map that translates batch indices (provided by the training/validation/test
            loops) into pairs of start/end indices of evaluation results in the internal dataframe.
            Used primarily for debugging and assertions.
        _dataframe: internal dataframe that will store the 'intermediate' evaluation results for all
            traces. Each row in this dataframe will correspond to a trace from a gather. The columns
            will store attributes that are either the evaluation results themselves or values that
            are necessary to summarize metrics according to specific categories.
        segm_class_count: the number of segmentation classes that the model is built to predict. If
            we are evaluating the results from a regression-only model, the value of this variable
            will be `None`. Otherwise, it should match a supported class count defined in the FBP
            constants module (``hardpicks.data.fbp.constants``).
        segm_first_break_prob_threshold: the threshold used to convert segmentation score maps into
            arrays of first break label indices. A higher threshold will leave more traces without
            a predicted first break but will also avoid false positives before the real break.
        metrics_metamap: the map of metric metadata that specifies what to produce when summarizing
            evaluation results. The keys in this map correspond to the name of metrics to produce,
            and the values are the evaluation parameters (if any).
    """

    def __init__(
        self,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Parses hyperparams, sets up static attributes, and prepares internal state for ingestion."""
        required_hp_names = [
            "eval_metrics",  # list of evaluation metrics to use (with specific parameters if needed)
            "segm_class_count",  # note: this is a model hyperparam needed here for preprocessing
            "segm_first_break_prob_threshold",  # threshold used to convert segm maps to fb indices
        ]
        check_hp(names=required_hp_names, hps=hyper_params, allow_extra=True)

        if "offset_buckets" in hyper_params:
            logger.warning("Found deprecated argument 'offset_buckets' passed to evaluator (ignored)")
        if "segm_confusion_dist" in hyper_params:
            logger.warning("Found deprecated argument 'segm_confusion_dist' passed to evaluator (ignored)")

        self.origin_id_map = {}  # will be used to avoid putting long/heavy strings in the dataframe
        self.seen_batch_idxs = {}  # will be used to make sure we don't see the same data twice
        self._dataframe = pd.DataFrame()  # will be refreshed w/ proper cols in reset below
        self.accumulated_trace_counts = 0
        self.list_batch_dataframes = []

        self.segm_class_count = hyper_params["segm_class_count"]
        self.segm_first_break_prob_threshold = hyper_params["segm_first_break_prob_threshold"]

        # should we extract the model's probability for the selected FBP?
        self.extract_fbp_probability = self.segm_class_count is not None

        # parse the list of metrics that should be computed each time we summarize the results
        self.metrics_metamap = self._init_metrics(hyper_params["eval_metrics"])

        # make a backup copy of the internal hyper parameters we need to reinstantiate later
        self.init_hyper_params = copy.deepcopy({key: hyper_params[key] for key in required_hp_names})

        # finally, put the evaluator into a proper state, ready for ingestion
        self.reset()

    def _init_metrics(
        self,
        metrics_array: typing.Iterable[typing.Dict],
    ) -> typing.Dict[typing.AnyStr, typing.Tuple[typing.AnyStr, typing.Dict]]:
        """Returns a map of metric metadata and segm/regr evaluation flags.

        This function will parse the list of metrics that are requested as the summarization results.
        While parsing this list, we will determine whether we need to produce the intermediate
        evaluation results for segmentation, regression, or both.

        The resulting metric map will be used in the ``_summarize_dataframe`` function where the
        final metrics will be computed from the intermediate results stored in the dataframe.

        Args:
            metrics_array: the array of metric definition dictionaries read from the config file. The
                definitions should contain two things: a type (under the ``metric_type`` key), and
                optional parameters for that metric (under the ``metric_params`` key).

        Returns:
            The map of metric metadata. In this map, keys correspond to unique metric names (derived
            from the metric type and any parameters, if necessary) and values correspond to the
            parameters required for evaluation.
        """
        assert all([isinstance(metric_definition, dict) for metric_definition in metrics_array])
        metrics_metamap = {}
        for metric_definition in metrics_array:
            metric_type = metric_definition["metric_type"]
            if metric_type not in SUPPORTED_METRICS:
                logger.warning(f"unknown metric type: {metric_type}. It will be assumed that this is a "
                               f"restart from an older checkpoint. This metric will be ignored.")
                continue

            assert metric_type not in SUPPORTED_SEGMENTATION_METRICS or self.segm_class_count is not None, \
                f"cannot use '{metric_type}' metric since model does not support segmentation"
            metric_params = metric_definition.get("metric_params", {})
            if metric_type == "HitRate":
                assert "buffer_size_px" in metric_params and metric_params["buffer_size_px"] >= 1, \
                    "missing or invalid hit rate buffer size in params (should be >=1 px)"
                metric_name = f"HitRate{metric_params['buffer_size_px']}px"
            elif metric_type in [
                "GatherCoverage", "MeanAbsoluteError", "RootMeanSquaredError", "MeanBiasError",
            ]:
                assert not metric_params, f"metric type '{metric_type}' does not expect any parameters"
                metric_name = metric_type
            else:  # pragma: no cover
                raise NotImplementedError
            assert metric_name not in metrics_metamap, "cannot have overlapping metric names in evaluator"
            metrics_metamap[metric_name] = (metric_type, metric_params)
        return metrics_metamap

    @staticmethod
    def _get_segm_coverage_array(
        target_idxs_map: np.ndarray,
        pred_first_break_labels: np.ndarray,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Returns the first-break coverage flags array for the given prediction/target maps.

        This function assumes that the given arguments are the predictions/targets for an entire
        gather (i.e. an array-of-traces), and that we want the coverage flags returned for each
        trace individually.

        The implementation below also assumes that part of the given targets map can be filled with
        'dontcare' labels due to padding (on the vertical or horizontal axes). For those samples, the
        coverage flag will be False.

        Args:
             target_idxs_map: a map of target class indices (nb_traces x nb_samples_per_trace).
             pred_first_break_labels: a 1d array of first-break sample index labels (one per trace).

        Returns:
            A tuple of arrays that flag 1) all traces with a proper prediction; and 2) all traces
            that should have had a proper prediction.
        """
        # here, we just assume that the segmentation-map-to-regr-labels conversion will have filled
        # the prediction array with NaNs whenever the model could not produce a proper result
        valid_mask = utils.get_valid_class_mask(target_idxs_map)
        # we determine the 'real' sample count by checking the maximum number of valid samples
        sample_count = np.max(valid_mask.sum(axis=1))
        assert sample_count != 0, "The inferred sample count is 0. " \
                                  "This breaks the assumption that at least one trace is valid in the gather. " \
                                  "Review code!"

        # we create the mask of traces that 'should' have been given a proper prediction
        valid_trace_flags = valid_mask[:, :sample_count].all(axis=1)
        # finally, let's compute the mask of traces that have been given a proper prediction
        good_pred_flags = np.logical_and(
            utils.get_valid_labels_mask(pred_first_break_labels),
            # we intersect with the mask of traces with available annotations
            # (this converts the original gather coverage into a 'hit rate at infinite threshold')
            valid_trace_flags,
        )
        return good_pred_flags, valid_trace_flags

    @profile
    def _get_segm_metrics_arrays(
        self,
        batch: typing.Any,
        batched_pred_first_breaks: torch.Tensor,
        good_traces_per_gather: typing.Sequence[typing.List[int]],
    ) -> typing.Dict[typing.AnyStr, np.ndarray]:
        """Returns the intermediate (trace-wise) evaluation results for segmentation predictions.

        This function will return arrays of binary flags indicating whether at least one sample was
        annotated as a probable first break in each trace. The evaluation results will be returned
        as a dictionary of arrays that can be easily converted into columns inside a dataframe.

        Args:
            batch: the entire minibatch dictionary loaded and provided by the data loader.
            batched_pred_first_breaks: the tensor of predicted first break label idxs produced by the model.
            good_traces_per_gather: the list of valid trace indices for each gather in the minibatch.
        """
        # first, unpack the inputs and prepare everything for processing on CPU w/ numpy
        batched_pred_first_breaks = batched_pred_first_breaks.cpu().numpy()
        assert len(batched_pred_first_breaks) == batch["batch_size"]
        assert len(good_traces_per_gather) == batch["batch_size"]
        assert "segmentation_mask" in batch, "forgot to generate the segmentation masks in preproc?"
        batched_target_idxs_map = batch["segmentation_mask"].cpu().numpy()
        assert len(batched_target_idxs_map) == batch["batch_size"]

        # initialize all the arrays where we will store the intermediate eval results
        tot_trace_count = sum([len(trace_idxs) for trace_idxs in good_traces_per_gather])
        metrics_arrays = {
            "GatherCoverage": np.zeros((tot_trace_count,), dtype=np.bool8),
            "ExpectedCoverage": np.zeros((tot_trace_count,), dtype=np.bool8),
        }

        # next, loop over all gathers in the batch and extract 'intermediate' evaluation results
        array_offset = 0
        for gather_idx in range(batch["batch_size"]):
            valid_trace_idxs = good_traces_per_gather[gather_idx]
            trace_count = len(valid_trace_idxs)
            array_start, array_end = array_offset, array_offset + trace_count
            pred_first_breaks = batched_pred_first_breaks[gather_idx, valid_trace_idxs]
            target_idxs_map = batched_target_idxs_map[gather_idx, valid_trace_idxs]
            coverage_array, expected_coverage_array = \
                self._get_segm_coverage_array(target_idxs_map, pred_first_breaks)
            assert len(coverage_array) == trace_count
            metrics_arrays["GatherCoverage"][array_start:array_end] = coverage_array
            assert len(expected_coverage_array) == trace_count
            metrics_arrays["ExpectedCoverage"][array_start:array_end] = expected_coverage_array
            array_offset += trace_count
        assert array_offset == tot_trace_count
        return metrics_arrays

    @staticmethod
    def _get_regr_error_array(
        batch: typing.Dict[typing.AnyStr, typing.Any],
        batched_pred_first_breaks: torch.Tensor,
        good_traces_per_gather: typing.Sequence[typing.List[int]],
    ) -> np.ndarray:
        """Returns an array of signed error values (one per trace) based on predictions.

        The errors are defined as the distance between the index of the trace sample labeled
        as the 'first break' and the index of the predicted first break. The predictions passed
        to this function are assumed to be first break indices, and NOT a map of class scores.

        If we ever want to evaluate prediction accuracy at a resolution smaller than the trace
        sample size, we'll have to rewrite this function to measure the distance in milliseconds
        instead of indices.

        Args:
            batch: the entire minibatch dictionary loaded and provided by the data loader.
            batched_pred_first_breaks: the tensor of predicted first break labels produced by the model.
            good_traces_per_gather: the list of valid trace indices for each gather in the minibatch.
        """
        # first, unpack the inputs and prepare everything for processing on CPU w/ numpy
        batched_pred_first_breaks = batched_pred_first_breaks.cpu().numpy()
        assert len(batched_pred_first_breaks) == batch["batch_size"]
        assert len(good_traces_per_gather) == batch["batch_size"]
        batched_target_first_breaks = batch["first_break_labels"].cpu().numpy()
        assert len(batched_target_first_breaks) == batch["batch_size"]
        assert batched_pred_first_breaks.shape == batched_target_first_breaks.shape, \
            "unexpected shape mismatch between pred/target tensors;\n" \
            f"\t{batched_pred_first_breaks.shape} vs {batched_target_first_breaks.shape}\n" \
            "... we're supposed to have regression results in here!"

        # initialize the array where we will store the intermediate eval results (i.e. regr errors)
        tot_trace_count = sum([len(trace_idxs) for trace_idxs in good_traces_per_gather])
        error_array = np.zeros((tot_trace_count,), dtype=np.float32)

        # next, loop over all gathers in the batch and extract the regression errors
        array_offset = 0
        for gather_idx in range(batch["batch_size"]):
            valid_trace_idxs = good_traces_per_gather[gather_idx]
            trace_count = len(valid_trace_idxs)
            array_start, array_end = array_offset, array_offset + trace_count
            pred_first_breaks = batched_pred_first_breaks[gather_idx, valid_trace_idxs]
            target_first_breaks = batched_target_first_breaks[gather_idx, valid_trace_idxs]
            valid_labels_mask = utils.get_valid_labels_mask(target_first_breaks)
            label_errors = (pred_first_breaks - target_first_breaks).astype(np.float32)
            error_array[array_start:array_end] = np.where(
                valid_labels_mask, label_errors, float("nan")
            )
            array_offset += trace_count
        assert array_offset == tot_trace_count
        return error_array

    @profile
    def ingest(
        self,
        batch: typing.Any,
        batch_idx: int,
        raw_preds: torch.Tensor,
    ) -> typing.Dict[typing.AnyStr, float]:
        """Ingests the batch data and the predictions from the model.

        Each time this function is called, it will record the tracewise evaluation results
        inside a dataframe. These results will be summarized and returned, but also appended
        to the internal dataframe for the end-of-epoch summarization.

        Arguments:
            batch: the minibatch provided by the data loader that contains all inputs/targets.
            batch_idx: the index of this particular step (or batch) in the current epoch.
            raw_preds: the predictions made by the model based on the minibatch input data. Can be
                either class score maps (from a segmentation model) or regressed values directly.

        Returns:
            A dictionary of metric evaluation results (for the current batch only).
        """
        if not self.metrics_metamap:
            return {}  # shortcut: there's nothing to ingest
        # if we see a minibatch again, we could go replace the corresponding rows.. (why though?)
        assert batch_idx not in self.seen_batch_idxs, "we've seen this minibatch already!"
        assert len(batch["rec_ids"]) == len(batch["offset_distances"])  # first dim == batch size

        # convert segmentation score maps into a regression-like output (using threshold/max score)
        regr_preds, probabilities_of_fbp = utils.get_regr_preds_from_raw_preds(
            raw_preds=raw_preds,
            segm_class_count=self.segm_class_count,
            prob_threshold=self.segm_first_break_prob_threshold,
        )
        if self.extract_fbp_probability:
            assert probabilities_of_fbp is not None, "The array probabilities_of_fbp is None, which is inconsistent."

        # gather info on which traces will lead to new rows in our dataframe & on category attributes
        good_traces_per_gather, trace_rec_ids, trace_origin_ids = [], [], []
        trace_shot_ids, trace_gather_ids, trace_offsets, trace_preds, trace_probs = [], [], [], [], []
        batched_receiver_ids = batch["rec_ids"].cpu().numpy()
        batched_offset_distances = batch["offset_distances"].cpu().numpy()
        for gather_idx in range(batch["batch_size"]):
            receiver_ids = batched_receiver_ids[gather_idx]
            # the 'good trace idxs' are those that are not padded and that will be used for evaluation
            good_traces_mask = utils.get_valid_traces_mask(receiver_ids)
            good_trace_idxs = np.where(good_traces_mask)[0]
            trace_count = len(good_trace_idxs)
            good_traces_per_gather.append(good_trace_idxs)
            # we also keep track of receiver IDs, origin IDs, shot IDs, and offsets for categorization...
            trace_rec_ids.extend(receiver_ids[good_trace_idxs])
            # for the origin (site name), if this gather is from a new one, add it to the map
            gather_origin_name = batch["origin"][gather_idx]
            if gather_origin_name not in self.origin_id_map:
                # we'll give this new site a whole new unique ID based on the total site count so far
                self.origin_id_map[gather_origin_name] = len(self.origin_id_map)
            # we know that all traces in a gather always come from the same site, so extend-copy it
            trace_origin_ids.extend([self.origin_id_map[gather_origin_name]] * trace_count)
            # same thing for shots, all traces in a gather always have the same shot/gather ids
            trace_shot_ids.extend([int(batch["shot_id"][gather_idx])] * trace_count)
            trace_gather_ids.extend([int(batch["gather_id"][gather_idx])] * trace_count)
            # remember: there are multiple offset channels, and the first one is the real one we need
            offset_distances = batched_offset_distances[gather_idx, good_trace_idxs, 0]
            trace_offsets.extend(offset_distances)
            # finally, update the array of raw predictions
            trace_preds.extend(regr_preds[gather_idx, good_trace_idxs].cpu().numpy())
            if self.extract_fbp_probability:
                trace_probs.extend(probabilities_of_fbp[gather_idx, good_trace_idxs].cpu().numpy())

        tot_trace_count = sum([len(good_trace_idxs) for good_trace_idxs in good_traces_per_gather])

        self.seen_batch_idxs[batch_idx] = \
            (self.accumulated_trace_counts, self.accumulated_trace_counts + tot_trace_count)

        # next, merge everything into a new dataframe segment with the intermediate evaluation results
        curr_dataframe = {
            "GatherId": pd.Series(data=trace_gather_ids, dtype="int"),
            "ShotId": pd.Series(data=trace_shot_ids, dtype="int"),
            "ReceiverId": pd.Series(data=trace_rec_ids, dtype="int"),
            "OriginId": pd.Series(data=trace_origin_ids, dtype="int"),
            "Offset": pd.Series(data=trace_offsets, dtype="float"),
            "Predictions": pd.Series(data=trace_preds, dtype="int"),
        }

        if self.extract_fbp_probability:
            curr_dataframe["Probabilities"] = pd.Series(data=trace_probs, dtype="float")

        use_segm_eval = \
            any([m[0] in SUPPORTED_SEGMENTATION_METRICS for m in self.metrics_metamap.values()])
        if use_segm_eval:
            # get the intermediate segmentation evaluation results & add them to the dataframe map
            eval_arrays = self._get_segm_metrics_arrays(
                batch, regr_preds, good_traces_per_gather,
            )
            for col_name, array in eval_arrays.items():
                assert col_name not in curr_dataframe
                assert len(array) == tot_trace_count, "bad array length returned by regr evaluator"
                curr_dataframe[col_name] = pd.Series(data=array, dtype=None)  # auto-detect dtype
        use_regr_eval = \
            any([m[0] in SUPPORTED_REGRESSION_METRICS for m in self.metrics_metamap.values()])
        if use_regr_eval:
            # get the intermediate segmentation evaluation results & add them to the dataframe map
            error_array = self._get_regr_error_array(batch, regr_preds, good_traces_per_gather)
            assert len(error_array) == tot_trace_count, "bad array length returned by regr evaluator"
            curr_dataframe["Errors"] = pd.Series(data=error_array, dtype="float")
        curr_dataframe = pd.DataFrame(curr_dataframe)

        # append the new dataframe segment to the big one that's got every segment so far
        assert np.array_equal(self._dataframe.columns, curr_dataframe.columns)

        self.accumulated_trace_counts += len(curr_dataframe)
        self.list_batch_dataframes.append(curr_dataframe)

        if len(self.list_batch_dataframes) > 500:
            self.finalize()

        # and finally, return the metrics for the current batch based on the new segment only
        return self._summarize_dataframe(curr_dataframe)

    def _summarize_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> typing.Dict[typing.AnyStr, float]:
        """Returns the metric evaluation results for the provided dataframe.

        This function assumes that any kind of category-level row select has already happened, and
        that we just need to aggregate all the intermediate results stored in the dataframe into
        the metrics that the user requested.

        The final treatment required for each metric is handled separately based on the metric type
        and based on the parameters tied to each one. If a metric evaluation returns a NaN, it will
        be entirely missing from the output dictionary.

        Args:
            dataframe: the pandas DataFrame object to summarize into a bunch of metrics.

        Returns:
            A map that connects each metric (by name) to its summarized value.
        """
        output_metrics = {}
        for metric_name, (metric_type, metric_params) in self.metrics_metamap.items():
            if metric_type == "HitRate":
                # here, we threshold the absolute error value with a buffer to get the hit counts
                tot_count = dataframe["Errors"].count()
                hit_slice = dataframe["Errors"].abs() < metric_params["buffer_size_px"]
                hit_slice[dataframe["Errors"].isnull()] = np.nan
                hit_count = hit_slice.sum()
                if tot_count > 0:
                    assert hit_count <= tot_count, "nan masking probably messed up above"
                    output_metrics[metric_name] = hit_count / tot_count
            elif metric_type in ["MeanAbsoluteError", "RootMeanSquaredError", "MeanBiasError"]:
                # for these, we apply dataframe-level ops on the error column before aggregating it
                if metric_type == "MeanAbsoluteError":
                    metric_val = dataframe["Errors"].abs().mean()
                elif metric_type == "RootMeanSquaredError":
                    metric_val = np.sqrt((dataframe["Errors"].abs() ** 2).mean())
                else:  # metric_type == "MeanBiasError":
                    metric_val = dataframe["Errors"].mean()
                if not np.isnan(metric_val):
                    output_metrics[metric_name] = metric_val
            elif metric_type == "GatherCoverage":
                if "ExpectedCoverage" not in dataframe.columns:
                    logger.warning(
                        "using old dataframe without expected coverage arrays; the gather "
                        "coverage will not correspond to the 'annotated trace' coverage, but "
                        "instead the coverage for all traces (without respect to which are labeled)"
                    )
                    output_metrics[metric_name] = float(dataframe["GatherCoverage"].mean())
                else:
                    coverage_count = dataframe["GatherCoverage"].sum()
                    expected_coverage_count = dataframe["ExpectedCoverage"].sum()

                    if expected_coverage_count > 0:
                        output_metrics[metric_name] = float(coverage_count) / float(expected_coverage_count)
                    else:
                        raise ValueError(f"The expected coverage count is {expected_coverage_count}. "
                                         "The assumption that all gathers have at least one valid trace is broken."
                                         f"Something is wrong, review code! The ingested dataframe is\n{dataframe}")
            else:
                raise NotImplementedError
        return output_metrics

    def summarize(
        self,
        category_name: typing.Optional[typing.AnyStr] = None,
    ) -> typing.Dict[typing.AnyStr, float]:
        """Returns the metric evaluation results for all recorded predictions.

        If no category name is provided, the returned metrics will be global aggregates.
        Otherwise, it will return a category-specific aggregation of metrics.

        Arguments:
            category_name: the name of the evaluation category to summarize.

        Returns:
            A dictionary of metric evaluation results.
        """
        self.finalize()
        selected_dataframe = None
        if category_name is not None:
            if category_name in self.origin_id_map:
                origin_idx = self.origin_id_map[category_name]
                matching_rows = self._dataframe["OriginId"] == origin_idx
            else:
                raise NotImplementedError
            if sum(matching_rows):
                selected_dataframe = self._dataframe[matching_rows]
        else:
            selected_dataframe = self._dataframe
        if selected_dataframe is not None and len(selected_dataframe):
            return self._summarize_dataframe(selected_dataframe)
        return {}

    def get_categories(self) -> typing.Sequence[typing.AnyStr]:
        """Returns the categories supported by this evaluator.

        Here, the categories are the origin (i.e. site names).

        Note that calling this function before the end of the epoch can result in a lower number
        of categories being returned as the 'origins' are based on uniques found in the dataframe.
        """
        origins = list(self.origin_id_map.keys())
        if len(origins) == 1:
            origins = []  # if there's only one site, skip the category, it's useless
        return origins

    def reset(self):
        """Resets the internal state of the evaluator object (useful when starting a new epoch)."""
        # the columns in our dataframe are the IDs used to uniquely identify/categorize the trace
        columns = {
            "GatherId": pd.Series([], dtype="int"),  # comes from the dataset parser directly
            "ShotId": pd.Series([], dtype="int"),  # comes from the dataset parser directly
            "ReceiverId": pd.Series([], dtype="int"),  # comes from the dataset parser directly
            "OriginId": pd.Series([], dtype="int"),  # holds the id of the trace's dataset
            "Offset": pd.Series([], dtype="float"),  # holds the raw rec-shot offsets from the parser
            "Predictions": pd.Series([], dtype="int"),  # holds the predicted first break indices
        }

        if self.extract_fbp_probability:
            columns["Probabilities"] = pd.Series(data=[], dtype="float")

        # ... plus the actual columns that hold the tracewise evaluation results
        use_segm_eval = \
            any([m[0] in SUPPORTED_SEGMENTATION_METRICS for m in self.metrics_metamap.values()])
        if use_segm_eval:
            columns["GatherCoverage"] = pd.Series([], dtype="bool")
            columns["ExpectedCoverage"] = pd.Series([], dtype="bool")
        use_regr_eval = \
            any([m[0] in SUPPORTED_REGRESSION_METRICS for m in self.metrics_metamap.values()])
        if use_regr_eval:
            columns["Errors"] = pd.Series([], dtype="float")
        self._dataframe = pd.DataFrame(columns=columns)
        self.list_batch_dataframes = []
        self.accumulated_trace_counts = 0
        self.origin_id_map = {}  # reset the origin id map too (we'll refill it every epoch)
        self.seen_batch_idxs = {}  # reset the seen batch idxs map (same logic)

    def dump(self, path: typing.AnyStr):
        """Dumps the internal state of the evaluator object at the given location."""
        self.finalize()
        with open(path, "wb") as fd:
            pickle.dump(dict(
                init_hyper_params=self.init_hyper_params,
                origin_id_map=self.origin_id_map,
                seen_batch_idxs=self.seen_batch_idxs,
                accumulated_trace_counts=self.accumulated_trace_counts,
                dataframe=self._dataframe,
            ), fd)

    @staticmethod
    def load(path: typing.AnyStr) -> "FBPEvaluator":
        """Loads the evaluator as previously dumped at the given location."""
        with open(path, "rb") as fd:
            attribs = pickle.load(fd)
        assert "init_hyper_params" in attribs, \
            "missing hparams dict in eval state dump; maybe the dump is too old?"
        eval = FBPEvaluator(attribs["init_hyper_params"])
        eval.origin_id_map = attribs["origin_id_map"]
        eval.seen_batch_idxs = attribs["seen_batch_idxs"]
        eval.accumulated_trace_counts = attribs["accumulated_trace_counts"]
        eval._dataframe = attribs["dataframe"]

        if eval.extract_fbp_probability:
            if 'Probabilities' not in eval._dataframe:
                logger.warning("The Evaluator parameters indicate that it should keep track of FBP probabilities,"
                               "but this column is absent from the pickle dataframe. Loading as is, assuming this is "
                               "a legacy pickle.")

        return eval

    def finalize(self):
        """Accumulate the step dataframes into one summary dataframe."""
        if len(self.list_batch_dataframes) > 0:
            self._dataframe = pd.concat([
                self._dataframe, pd.concat(self.list_batch_dataframes, ignore_index=True)
            ])
            self.list_batch_dataframes = []
