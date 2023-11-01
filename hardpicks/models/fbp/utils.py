"""FBP model utility functions module."""

import typing

import cv2 as cv
import numpy as np
import torch
import torch.utils.data

import hardpicks.data.fbp.constants as constants
import hardpicks.metrics.fbp.utils as metric_utils
import hardpicks.utils.draw_utils as draw_utils


def prepare_input_features(
    batch: typing.Any,
    use_dist_offsets: bool = False,
    use_first_break_prior: bool = False,
    augmentations: typing.Optional[typing.Iterable[typing.Callable]] = None,
) -> torch.Tensor:
    """Returns the 'input feature tensor' to forward through the model for FBP tasks."""
    assert "samples" in batch, "missing the batched 2D arrays that contain sampled amplitudes?"
    input_tensor = batch["samples"]
    assert input_tensor.ndim == 3  # should not have a channel dim yet
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    if use_dist_offsets:
        assert "offset_distances" in batch, \
            "missing 'offset_distances' field in gathers (is 'provide_offset_dists' turned on?)"
        offset_distances = batch["offset_distances"]
        assert offset_distances.ndim == 3
        assert offset_distances.shape[0] == input_tensor.shape[0]
        assert offset_distances.shape[1] == input_tensor.shape[2]
        assert offset_distances.shape[2] == 3
        offset_distances = torch.transpose(offset_distances, 1, 2).unsqueeze(-1)
        input_tensor = torch.cat([
            input_tensor,
            offset_distances.repeat(1, 1, 1, input_tensor.shape[-1]),
        ], dim=1).type(input_tensor.dtype)
    if use_first_break_prior:
        assert "first_break_prior" in batch, \
            "missing 'first_break_prior' in gathers (is 'generate_first_break_prior_masks' turned on?)"
        first_break_prior = batch["first_break_prior"]
        assert first_break_prior.ndim == 3
        assert first_break_prior.shape[0] == input_tensor.shape[0]
        assert first_break_prior.shape[1] == input_tensor.shape[2]
        assert first_break_prior.shape[2] == input_tensor.shape[3]
        input_tensor = torch.cat([
            input_tensor,
            first_break_prior.unsqueeze(1),
        ], dim=1).type(input_tensor.dtype)
    # TODO: if we want to concatenate more features into the input tensor, we'd do it here
    if augmentations:
        for augop in augmentations:
            input_tensor = augop(input_tensor)
    return input_tensor


def generate_pred_image(
    batch: typing.Dict,
    raw_preds: typing.Optional[torch.Tensor],
    batch_gather_idx: int,
    segm_class_count: int,
    segm_first_break_prob_threshold: float,
    draw_gt: bool = True,
    draw_prior: bool = False,
    draw_prob_heatmap: bool = True,
):  # pragma: no cover
    """Generates a displayable 2d image of the samples map with the first breaks drawn as points."""
    # first, get the raw gather image without anything extra on top
    gather_image = draw_utils.generate_gather_image_from_batch(batch, batch_gather_idx, draw_prior)

    # now, draw the pred+target first breaks (wherever available) on top of that image
    valid_trace_mask = metric_utils.get_valid_traces_mask(batch["rec_ids"][batch_gather_idx]).cpu().numpy()
    target_first_break_labels = batch["first_break_labels"][batch_gather_idx].cpu().numpy()
    bad_first_break_mask = batch["bad_first_breaks_mask"][batch_gather_idx].cpu().numpy()
    if "outlier_first_breaks_mask" in batch:
        outlier_first_break_mask = batch["outlier_first_breaks_mask"][batch_gather_idx].cpu().numpy()
    else:
        # if we dont have an outlier mask, just use the bad pick mask, it won't change anything
        outlier_first_break_mask = batch["bad_first_breaks_mask"][batch_gather_idx].cpu().numpy()
    pred_first_break_labels = None
    if raw_preds is not None:
        regr_preds, _ = metric_utils.get_regr_preds_from_raw_preds(
            raw_preds=raw_preds,
            segm_class_count=segm_class_count,
            prob_threshold=segm_first_break_prob_threshold,
        )
        pred_first_break_labels = regr_preds[batch_gather_idx].cpu().numpy()
    target_image = draw_utils.draw_first_break_points(
        image=np.copy(gather_image),
        pred_first_break_labels=pred_first_break_labels,
        target_first_break_labels=target_first_break_labels if draw_gt else None,
        valid_trace_mask=valid_trace_mask,
        bad_first_break_mask=bad_first_break_mask,
        outlier_first_break_mask=outlier_first_break_mask,
    )
    target_image = cv.resize(target_image, dsize=(-1, -1), fx=2, fy=2, interpolation=cv.INTER_NEAREST)

    if raw_preds is not None and draw_prob_heatmap:
        target_image = draw_utils.draw_big_text_at_top_right(target_image, "preds vs groundtruth")
        # next, generate a heatmap of the scores of the first break (or after first break) class
        pred_fb_prob = metric_utils.get_first_break_class_prob_map(
            raw_preds=raw_preds,
            segm_class_count=segm_class_count,
        ).cpu().numpy()[batch_gather_idx]
        # make sure the heatmap normalization sticks to the real [0, 1] probability range
        pred_fb_prob[0, -1], pred_fb_prob[-1, -1] = 0.0, 1.0
        pred_fb_heatmap = draw_utils.add_heatmap_on_base_image(gather_image, pred_fb_prob)
        pred_fb_heatmap = cv.resize(pred_fb_heatmap, dsize=(-1, -1), fx=2, fy=2, interpolation=cv.INTER_NEAREST)
        pred_class_name = constants.SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP[segm_class_count][-1]
        pred_fb_heatmap = draw_utils.draw_big_text_at_top_right(pred_fb_heatmap, f"'{pred_class_name}' prob")

        # finally, stack the two images into one big output
        image = cv.vconcat([target_image, pred_fb_heatmap])
    else:
        image = target_image

    return image
