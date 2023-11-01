"""This module contains utility functions related to FBP evaluation.

This is just meant to de-clutter the FBP evaluator class from static functions.
"""

import typing

import numpy as np
import torch

from hardpicks.data.fbp.constants import (
    BAD_FIRST_BREAK_PICK_INDEX,
    BAD_OR_PADDED_ELEMENT_ID,
    SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP,
)
from hardpicks.models.constants import DONTCARE_SEGM_MASK_LABEL


def get_class_idx(
    segm_class_count: int, class_name: typing.AnyStr,
) -> int:  # pragma: no cover
    """Returns the class index (i.e. label value used in masks) for a particular class name."""
    class_names = SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP[segm_class_count]
    return class_names.index(class_name)


def get_class_names(
    segm_class_count: int,
) -> typing.List[typing.AnyStr]:  # pragma: no cover
    """Returns the class names list for the current segmentation task."""
    class_names = SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP[segm_class_count]
    if segm_class_count == 1:
        # special case: there's always two classes minimum (binary classif), but we keep one
        class_names = [class_names[-1]]
    return class_names


def get_intervals_from_break_points(
    break_points: typing.Iterable[typing.Tuple[int, float]],
) -> typing.List[typing.Tuple[int, int]]:
    """Returns a list of value intervals (minimum, maximum) based on a list of break points.

    This function is made to work with strictly-positive break points, and returns min-max pairs
    that start at 0 and end at +inf. If no break point is given in the provided list, the function
    will return a single interval (0, +inf). With each extra break point, it will add a new interval
    to the output.

    Args:
        break_points: a list of 'breaking points' (limits between intervals). Can be empty.
    """
    latest_interval_end = 0
    output_intervals = []
    if break_points is not None and break_points:
        assert all([np.isscalar(break_point) for break_point in break_points])
        for break_point in break_points:
            assert (
                break_point > latest_interval_end
            ), "break points must be sorted and non-overlapping"
            output_intervals.append((float(latest_interval_end), float(break_point)))
            latest_interval_end = break_point
    output_intervals.append((float(latest_interval_end), float("inf")))
    return output_intervals


def assign_interval_ids(
    val_array: np.ndarray, interval_pairs: typing.Iterable[typing.Tuple[float, float]],
) -> np.ndarray:
    """Returns the list of interval indices in which each value of the given array falls."""
    bucket_ids = np.zeros_like(val_array, dtype=np.int32)
    for (
        bucket
    ) in interval_pairs:  # iteratively-add-compare the upper bound to get the id
        bucket_ids += (val_array > bucket[1]).astype(np.int32)
    return bucket_ids


def get_valid_traces_mask(receiver_ids: np.ndarray,) -> np.ndarray:  # pragma: no cover
    """Returns a boolean mask that indicates whether the receiver IDs are valid or not."""
    # this utility function is used just to not get confused with the logical operator to apply...
    return receiver_ids != BAD_OR_PADDED_ELEMENT_ID


def get_valid_class_mask(class_idxs_map: np.ndarray,) -> np.ndarray:  # pragma: no cover
    """Returns a boolean mask that indicates whether the class idxs are valid or not."""
    # this utility function is used just to not get confused with the logical operator to apply...
    assert class_idxs_map.ndim >= 2, "this should be a map, but it's just an array?"
    # we don't actually check to see if the class idxs are out-of-bounds, that's a bit overkill
    return class_idxs_map != DONTCARE_SEGM_MASK_LABEL


def get_valid_labels_mask(
    sample_idxs_array: typing.Union[np.ndarray, torch.Tensor],
) -> typing.Union[np.ndarray, torch.Tensor]:  # pragma: no cover
    """Returns a boolean mask that indicates whether the sample idxs are valid or not."""
    # this utility function is used just to not get confused with the logical operator to apply...
    assert (
        sample_idxs_array.ndim >= 1
    ), "this should be an array, but it's just a scalar?"
    # we don't actually check the upper bound here (it's hard to do unless we get the sample count)
    return sample_idxs_array > BAD_FIRST_BREAK_PICK_INDEX


def get_first_break_class_prob_map(
    raw_preds: torch.Tensor, segm_class_count: int,
) -> torch.Tensor:
    """Returns the predicted probability map of the first break class in the raw array."""
    assert (
        segm_class_count in SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP
    ), "unexpected segmentation model class count (should be in supported map!)"
    actual_class_count = len(SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP[segm_class_count])
    assert (
        raw_preds.ndim == 4 and raw_preds.shape[1] == actual_class_count
    ), "unexpected segmentation model output shape; 2nd dim should be the class scores"
    # note: in the two-class setup (before/after first break), we use the 2nd class (after) here
    return torch.softmax(raw_preds, dim=1)[
        :, -1
    ]  # yep, it's easy, all setups use the same index


def get_regr_preds_from_raw_preds(
    raw_preds: torch.Tensor, segm_class_count: typing.Union[int, None], prob_threshold: float = 0.5,
) -> (torch.Tensor, typing.Union[torch.Tensor, None]):
    """Get regression predictions from raw predictions.

    This method produces the first break pick predictions as well as the probability of those
    predictions, if relevant.

    There are two branches to this method:

    1) Regression mode:
        The raw predictions are already regression values. This is signaled by setting the input variable
        "segm_class_count" to None. In this case, we simply return the raw predictions directly, and since
        "probabilities" are meaningless in this case, we return "None" for them.

    2) Classification mode:
        The raw predictions are the unormalized outputs of a classification model (i.e., the softmax has not yet
        been applied to turn the predictions into probabilities). The number of classes is indicated by
        segm_class_count.

    Args:
        raw_preds (torch.Tensor): raw model output. Either a regression directly, or class probabilities for each
                                  sample.
        segm_class_count (int or None): the number of segmentation classes. If "None", indicates that the
                                        input is already a regression result.
        prob_threshold (float): threshold above which a probability must be to produce a prediction. Ignored if
                                segm_class_count is None.

    Returns:
        first_break_picks (torch.Tensor): the values of the first break picks. Can be continuous numbers in
                                        regression mode, or the index of the first break pick in classification mode.
        probabilities (torch.Tensor or None): probabilities of selected first break indices in classification mode,
                                            or None in regression mode.
    """
    if segm_class_count is None:
        # Regression mode
        # we're not using prediction maps as the 'raw' format, return regression results as-is, and
        # None for probabilities since this concept is meaningless in this context.
        return raw_preds, None

    # Classification mode

    # The raw_preds tensor has dimensions:
    #   [batch size, number of classes, number of traces, number of sample per traces]

    # the maximum probability pixel/sample with first-break confidence >= threshold becomes the first break
    # NOTE: 0.5 is probably a bad pick for the 3-class setup, e.g. 0.3/0.3/0.4 will not give a FB

    # The first_break_probabilities tensor has dimensions:
    #   [batch size, number of traces, number of sample per traces]
    # -> we only track the "first break class", so the number of class dimension disappears.
    first_break_probabilities = get_first_break_class_prob_map(
        raw_preds, segm_class_count
    )

    # The fbp_indices tensor has dimensions:
    #   [batch size, number of traces]
    # -> We select the index of the first break for each trace.
    if segm_class_count == 1 or segm_class_count == 3:
        fbp_indices = _get_maximum_index_if_probability_above_threshold(
            first_break_probabilities, prob_threshold
        )
    else:
        fbp_indices = _get_first_index_where_probability_above_threshold(
            first_break_probabilities, prob_threshold
        )

    probabilities_of_predicted_fbp = _get_probabilities_of_fbp_indices(first_break_probabilities, fbp_indices)

    return fbp_indices, probabilities_of_predicted_fbp


def _get_probabilities_of_fbp_indices(fbp_probabilities: torch.Tensor, fbp_indices: torch.Tensor) -> torch.Tensor:
    """Get probabilities of first break pick indices.

    This method extracts the probability of the selected first break pick index, using NaN to
    indicate no-pick / invalid pick.

    It is assumed that the inputs are consistent, i.e., that the fbp_indices were selected based on
    probabilities. No check is performed to check this assumption.

    Args:
        fbp_probabilities (torch.Tensor): fbp probabilities for all samples
        fbp_indices (torch.Tensor): indices

    Returns:
        probabilities_of_selected_fbp (torch.Tensor): probabilities for selected fbp indices.
    """
    # The first_break_probabilities tensor has dimensions:
    #   [batch size, number of traces, number of sample per traces]
    # and the fbp_indices tensor has dimensions:
    #   [batch size, number of traces]
    probs = fbp_probabilities.gather(2, fbp_indices.to(torch.long).unsqueeze(-1)).squeeze(-1)
    valid_pick_mask = get_valid_labels_mask(fbp_indices)
    probs[~valid_pick_mask] = np.NaN

    return probs


def _get_maximum_index_if_probability_above_threshold(
    probabilities: torch.Tensor, prob_threshold: float = 0.5
):
    maximum_probabilities, indices_of_maximum_probabilities = torch.max(
        probabilities, dim=2
    )

    # note: if all pixels are < threshold confidence, this will return index=0
    # ... that will be perfect, because it'll be caught as an 'invalid'/'bad' pick downstream
    small_probability_mask = maximum_probabilities < prob_threshold
    indices_of_maximum_probabilities[small_probability_mask] = 0

    return indices_of_maximum_probabilities


def _get_first_index_where_probability_above_threshold(
    probabilities: torch.Tensor, prob_threshold: float = 0.5
):
    # the first pixel/sample whose first-break confidence goes >= threshold becomes the first break
    # NOTE: 0.5 is probably a bad pick for the 3-class setup, e.g. 0.3/0.3/0.4 will not give a FB
    first_break_mask = probabilities >= prob_threshold
    # note: if all pixels are < threshold confidence, this will return index=0
    # ... that will be perfect, because it'll be caught as an 'invalid'/'bad' pick downstream

    # The pytorch docs say:
    #   ``If there are multiple maximal values then the indices of the first maximal value are returned."
    # However, it is a little dangerous to rely on this behavior, and it can change unexpectedly from one version
    # to another. We must thus make sure there is a unique first break pick index so we can extract it consistently.

    # If there are no probabilities above prob_threshold, then the sum along a trace will be zero.
    first_break_is_present = first_break_mask.sum(dim=2).bool()

    # If there are probabilities above prob_threshold, then the cumsum along a trace will be one for the first.
    first_break_cumulative_sum = first_break_mask.cumsum(dim=2)

    # The first "first break" will have a value of zero, everything else will be higher value.
    first_break_identifier_array = torch.abs(first_break_cumulative_sum - 1)
    first_break_indices = torch.argmin(first_break_identifier_array, dim=2)

    # multiply by the "is_present" mask, so we return zero if a first break is not present.
    return first_break_is_present * first_break_indices


def compute_iou_from_classif_counts(
    true_positives: int, false_positives: int, false_negatives: int,
) -> float:
    """Computes the Intersection over Union (IoU) score based on TP/FP/FN counts."""
    union = true_positives + false_positives + false_negatives
    if union == 0:
        return float("nan")
    return true_positives / union
