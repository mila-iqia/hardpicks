"""Contains all transformation operations that can be applied to gathers.

The gathers transformed by these operations are assumed to be loaded via the
``gather_parser'' module, meaning we'll use a bunch of the hard-coded attribute
names here to fetch stuff from the given dictionaries.
"""
import typing

import numpy as np
import scipy.signal
import scipy.stats

import hardpicks.data.fbp.collate as collate
import hardpicks.data.fbp.constants as consts
import hardpicks.models.constants as generic_consts

GatherType = typing.Dict[typing.AnyStr, typing.Any]

SPECIAL_GATHER_ATTRIB_NAMES = [
    # these can't be default-transformed in the ops below (they're not 1D arrays)
    *collate.get_fields_to_double_pad(),
    "offset_distances",  # also cheating a bit here, we need to make sure that name exists...
]


def _drop_trace_idxs(
    gather: GatherType,
    drop_idxs: typing.Sequence[int],
):
    """Drops a particular trace from a gather. Utility function used by `drop_traces`."""
    assert len(drop_idxs) > 0
    drop_idxs = np.sort(drop_idxs).tolist()
    assert all([0 <= idx < gather["trace_count"] for idx in drop_idxs])
    # we can drop traces directly in each array except for 'offset distances'...
    for attrib_name, _ in collate.get_fields_to_pad():
        if attrib_name in gather and attrib_name != "offset_distances":
            gather[attrib_name] = np.delete(gather[attrib_name], drop_idxs, axis=0)
    # these take a bit more work... one pack at a time! (need to patch rec-to-rec dists both ways)
    # (no, this is not 100% ideal for non-collinear receivers, but data augmentation ain't perfect
    if "offset_distances" in gather:
        offsets = gather["offset_distances"]
        for idx_to_remove in reversed(drop_idxs):
            if idx_to_remove > 0:
                # transfer over current "dist-to-next" to previous receiver's "dist-to-next"
                offsets[idx_to_remove - 1, 1] += offsets[idx_to_remove, 1]
            if idx_to_remove < gather["trace_count"] - 1:
                # transfer over current "dist-to-prev" to next receiver's "dist-to-prev"
                offsets[idx_to_remove + 1, 2] += offsets[idx_to_remove, 2]
        # now, drop the actual idxs we don't want...
        offsets = np.delete(offsets, drop_idxs, axis=0)
        # patch the first/last dists to init values (zeros)
        offsets[-1, 1], offsets[0, 2] = 0, 0
        assert len(offsets) == gather["trace_count"] - len(drop_idxs)
        gather["offset_distances"] = offsets
    gather["trace_count"] = gather["trace_count"] - len(drop_idxs)


def drop_traces(
    gather: GatherType,
    drop_count: int,  # number of traces to drop from the gather
    drop_bad_picks_first: bool,  # toggles whether to drop bad picks first (across the gather)
    drop_edges_next: bool,  # toggles whether once bad picks have been dropped whether to drop edges
):
    """Drops some of the traces inside the provided gather (randomly, bad-traces-first, or edge-first)."""
    # note: val/test metrics should never be computed on gathers with dropped traces!
    assert 0 <= drop_count < gather["trace_count"]
    if drop_count == 0:
        return
    to_drop = []
    if drop_bad_picks_first:
        # dropping bad picks first will allow us to keep useful traces as long as possible for training
        bad_pick_idxs = np.where(gather["bad_first_breaks_mask"])[0]
        new_idxs_count = min(len(bad_pick_idxs), drop_count)
        to_drop.extend(np.random.choice(bad_pick_idxs, new_idxs_count, replace=False).tolist())
    # here, we compute the number of indices to drop (still) and make the list of available ones
    must_still_drop = drop_count - len(to_drop)
    remaining_idxs = [idx for idx in range(gather["trace_count"]) if idx not in to_drop]
    assert len(remaining_idxs) > must_still_drop
    # if we decide to drop edges before randomly dropping samples, just alternate front-and-back
    if drop_edges_next:
        while must_still_drop:
            to_drop.append(remaining_idxs.pop(0))
            must_still_drop -= 1
            if not must_still_drop:
                break
            to_drop.append(remaining_idxs.pop())
            must_still_drop -= 1
    # otherwise, we'll just directly sample the final indices of traces we still need to drop
    if must_still_drop:
        to_drop.extend(np.random.choice(remaining_idxs, must_still_drop, replace=False).tolist())
    # ... pass these off to the other func, and we're done!
    _drop_trace_idxs(gather, to_drop)


def pad_traces(
    gather: GatherType,
    prepad_trace_count: int,
    postpad_trace_count: int,
):
    """Adds a specified amount of (trace-wise) padding at the beginning/end of a gather."""
    assert prepad_trace_count >= 0 and postpad_trace_count >= 0
    if prepad_trace_count == 0 and postpad_trace_count == 0:
        return
    pad_value_map = {val[0]: val[1] for val in collate.get_fields_to_pad()}
    # first, pad the flat attributes (should all be 1d arrays)
    for attrib_name, attrib_pad_val in collate.get_fields_to_pad():
        if attrib_name in gather and attrib_name not in SPECIAL_GATHER_ATTRIB_NAMES:
            prepad_shape = (prepad_trace_count, *gather[attrib_name][0].shape)
            prepad = np.full(prepad_shape, pad_value_map[attrib_name], gather[attrib_name].dtype)
            postpad_shape = (postpad_trace_count, *gather[attrib_name][0].shape)
            postpad = np.full(postpad_shape, pad_value_map[attrib_name], gather[attrib_name].dtype)
            gather[attrib_name] = np.concatenate([prepad, gather[attrib_name], postpad])
    # then, pad the offset distances (this is a 3-ch array, if it is defined at all)
    if "offset_distances" in gather and gather["offset_distances"] is not None:
        tar_dtype = gather["offset_distances"].dtype
        prepad = np.full((prepad_trace_count, 3), pad_value_map["offset_distances"], tar_dtype)
        postpad = np.full((postpad_trace_count, 3), pad_value_map["offset_distances"], tar_dtype)
        gather["offset_distances"] = np.concatenate([
            np.asarray(prepad), gather["offset_distances"], np.asarray(postpad),
        ])
    # next, pad the samples array (this is a 2d array)
    n_samples = gather["sample_count"]
    tar_dtype = gather["samples"].dtype
    prepad = np.full((prepad_trace_count, n_samples), pad_value_map["samples"], tar_dtype)
    postpad = np.full((postpad_trace_count, n_samples), pad_value_map["samples"], tar_dtype)
    gather["samples"] = np.concatenate([prepad, gather["samples"], postpad])
    # finally, pad the segmentation mask array (this is a 2d array, if it is defined at all)
    if "segmentation_mask" in gather and gather["segmentation_mask"] is not None:
        raise AssertionError("based on order of operations in preproc, should never happen?")
    # as a bonus: let's adjust the trace count we keep in the gather to avoid collate issues
    gather["trace_count"] = prepad_trace_count + gather["trace_count"] + postpad_trace_count


def kill_traces(
    gather: GatherType,
    prob: float,
):
    """Kills traces (making amplitudes zero across all samples) with a given probability."""
    assert 0 < prob < 1
    trace_count = gather["trace_count"]
    kill_mask = np.random.choice([True, False], trace_count, p=[prob, 1 - prob])
    gather["samples"][kill_mask, :] = 0.


def _create_noise_patch(
    sample_count: int,  # number of samples (width) of the noise patch
    trace_count: int,  # number of traces (height) of the noise patch
    max_amplitude: float,  # maximum (absolute) amplitude of the noise inside the patch
    min_frequency: float,  # minimum noise frequency (in pixels/samples) inside the patch
    max_frequency: float,  # maximum noise frequency (in pixels/samples) inside the patch
    max_parallel_bands: int,  # maximum number of parallel traces that will share noise stats
):
    """Creates a 2D patch of noise based on a given set of parameters."""
    assert max_amplitude > 0
    assert min_frequency < max_frequency
    assert 0 <= max_parallel_bands
    patch = np.zeros((trace_count, sample_count))
    band_frequency, band_offset, max_band_size, band_idx = 0, 0, 0, 0
    for trace_idx in range(trace_count):
        if max_band_size == 0 or band_idx >= max_band_size:
            max_band_size = np.random.randint(max_parallel_bands) + 1
            band_frequency = np.random.randint(min_frequency, max_frequency)
            band_offset = np.random.randint(band_frequency)
            band_idx = 0
        else:
            # we add a bit of sample-wise freq/offset jitter to make it feel more natural...
            band_frequency = band_frequency + np.random.randint(2)
            band_offset = band_offset + np.random.randint(3) - 1
        patch[trace_idx, :] = \
            np.cos(((np.arange(sample_count) + band_offset) / band_frequency) * np.pi * 2)
        band_idx += 1
    return patch * max_amplitude


def add_noise_patch(
    gather: GatherType,
    prob: float,  # probability of applying a noise patch in this gather
    max_amplitude_factor: float = 1.0,  # maximum noise amplitude factor (wrt gather-wise max)
    min_sample_noise_frequency: int = 2,  # minimum sample-wise noise frequency inside the patch
    max_sample_noise_frequency: int = 50,  # maximum sample-wise noise frequency inside the patch
    max_trace_ratio_count: float = 0.15,  # maximum ratio of traces that the patch may cover
    sample_ratio_count: float = 1.0,  # ratio of samples that the patch will cover
    max_parallel_traces: int = 5,  # maximum number of parallel traces at a similar frequency/offset
    min_patch_sigma: float = 0.1,  # minimum patch smoothing factor
    max_patch_sigma: float = 1.0,  # maximum patch smoothing factor
):
    """Adds a noise 'patch' inside the gather."""
    assert 0 <= prob <= 1
    if np.isclose(prob, 0.0) or np.random.rand() >= prob:
        return
    # first, init all parameters we need to define the patch (location, frequency, size, ...)
    trace_count, sample_count = gather["trace_count"], gather["sample_count"]
    max_patch_trace_count = int(min(max(trace_count * max_trace_ratio_count, 1), trace_count))
    patch_trace_count = np.random.randint(max_patch_trace_count) + 1
    patch_sample_count = int(min(max(sample_count * sample_ratio_count, 1), sample_count))
    patch_center = np.random.randint(trace_count), np.random.randint(sample_count)
    max_real_ampl = np.abs(gather["samples"]).max()
    patch_ampl = np.random.random() * max_amplitude_factor * max_real_ampl
    assert min_sample_noise_frequency < max_sample_noise_frequency
    assert min_patch_sigma < max_patch_sigma
    patch_sigma = min_patch_sigma + np.random.random() * (max_patch_sigma - min_patch_sigma)
    assert 0 <= max_parallel_traces
    # then, create the patch noise, and apply it on the 2D array
    patch_noise = _create_noise_patch(
        sample_count=patch_sample_count,
        trace_count=patch_trace_count,
        max_amplitude=patch_ampl,
        min_frequency=min_sample_noise_frequency,
        max_frequency=max_sample_noise_frequency,
        max_parallel_bands=max_parallel_traces,
    )
    kernel_space = np.dstack((
        np.linspace(-1, 1, num=patch_trace_count, endpoint=True).reshape(-1, 1).repeat(patch_sample_count, axis=1),
        np.linspace(-1, 1, num=patch_sample_count, endpoint=True).reshape(1, -1).repeat(patch_trace_count, axis=0),
    ))
    kernel = scipy.stats.multivariate_normal(
        mean=[0., 0.],
        cov=[[patch_sigma, 0], [0, patch_sigma]],
    ).pdf(kernel_space)
    patch_noise *= (kernel.reshape(patch_noise.shape) / kernel.max())
    # uncomment code below to get an idea what the noise will look like...
    # import cv2 as cv
    # test = cv.normalize(patch_noise, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # test = cv.resize(test, (-1, -1), dst=None, fx=8, fy=8, interpolation=cv.INTER_NEAREST)
    # cv.imshow("test", test)
    # cv.waitKey(0)
    assert gather["samples"].shape == (trace_count, sample_count)
    gather_min_y = max(0, patch_center[0] - patch_trace_count // 2)
    gather_min_x = max(0, patch_center[1] - patch_sample_count // 2)
    gather_max_y = min(gather_min_y + patch_trace_count, trace_count)
    gather_max_x = min(gather_min_x + patch_sample_count, sample_count)
    assert gather_max_y - gather_min_y > 0 and gather_max_x - gather_min_x > 0
    gather["samples"][gather_min_y:gather_max_y, gather_min_x:gather_max_x] += \
        patch_noise[0:gather_max_y - gather_min_y, 0:gather_max_x - gather_min_x]


def crop_samples(
    gather: GatherType,
    max_samples: int,
):
    """Crops the samples (and segmentation mask) attribs to a maximum number of values."""
    # note: we will remove annotations if we make them go "out-of-range"!
    #       val/test metrics should never be computed on crops!
    # note2: only the 'label' annotations are modified, the timestamp annotations are kept intact
    assert max_samples > 0
    n_traces, n_samples = gather["trace_count"], gather["sample_count"]
    if max_samples >= n_samples:
        return
    assert gather["samples"].shape == (n_traces, n_samples)
    gather["samples"] = gather["samples"][:, :max_samples]
    if "segmentation_mask" in gather and gather["segmentation_mask"] is not None:
        raise AssertionError("based on order of operations in preproc, should never happen?")
    oob_mask = gather["first_break_labels"] >= max_samples
    new_bad_first_breaks_mask = np.logical_or(oob_mask, gather["bad_first_breaks_mask"])
    gather["bad_first_breaks_mask"] = new_bad_first_breaks_mask
    if "outlier_first_breaks_mask" in gather and gather["outlier_first_breaks_mask"] is not None:
        new_outlier_first_breaks_mask = np.logical_or(oob_mask, gather["outlier_first_breaks_mask"])
        gather["outlier_first_breaks_mask"] = new_outlier_first_breaks_mask
    gather["first_break_labels"][oob_mask] = consts.BAD_FIRST_BREAK_PICK_INDEX
    assert (gather["first_break_labels"] < max_samples).all()
    gather["sample_count"] = max_samples


def resample(
    gather: GatherType,
    target_sample_count: int,
):
    """Resamples the amplitude data inside the gather, rescaling all first break labels as needed."""
    # import this module here to avoid circular dependencies
    import hardpicks.data.fbp.trace_parser as trace_parser
    # note: here, we'll update the first break labels and leave timestamps and masks intact;
    #       val/test metrics should never be computed on resampled gathers!
    n_traces, n_samples = gather["trace_count"], gather["sample_count"]
    samples, fb_timestamps = gather["samples"], gather["first_break_timestamps"]
    assert fb_timestamps.ndim == 1 and len(fb_timestamps) == n_traces
    assert samples.shape == (n_traces, n_samples)
    if target_sample_count == n_samples:
        return
    resampl_ratio = target_sample_count / n_samples
    sample_rate_ms = gather["sample_rate_ms"]
    new_sample_rate_ms = sample_rate_ms / resampl_ratio
    fb_labels = gather["first_break_labels"]
    fb_labels = trace_parser.RawTraceDataset._get_first_break_indices(
        fb_timestamps.astype(np.float32),
        new_sample_rate_ms,
    ).astype(fb_labels.dtype)
    oob_mask = fb_labels >= target_sample_count
    new_bad_first_breaks_mask = np.logical_or(oob_mask, gather["bad_first_breaks_mask"])
    gather["bad_first_breaks_mask"] = new_bad_first_breaks_mask
    if "outlier_first_breaks_mask" in gather and gather["outlier_first_breaks_mask"] is not None:
        new_outlier_first_breaks_mask = np.logical_or(oob_mask, gather["outlier_first_breaks_mask"])
        gather["outlier_first_breaks_mask"] = new_outlier_first_breaks_mask
    fb_labels[oob_mask] = consts.BAD_FIRST_BREAK_PICK_INDEX
    assert (fb_labels < target_sample_count).all()
    gather["first_break_labels"] = fb_labels
    gather["samples"] = scipy.signal.resample(
        samples,
        num=target_sample_count,
        axis=1,
    )
    gather["sample_count"] = target_sample_count
    gather["sample_rate_ms"] = new_sample_rate_ms


def flip(
    gather: GatherType
):
    """Flips the gather (as well as all its IDs and other stuff) along the receiver-line-axis."""
    n_traces, n_samples = gather["trace_count"], gather["sample_count"]
    # first, flip the flat attributes (should all be 1d arrays)
    for attrib_name, attrib_pad_val in collate.get_fields_to_pad():
        if attrib_name in gather and attrib_name not in SPECIAL_GATHER_ATTRIB_NAMES:
            assert len(gather[attrib_name]) == n_traces
            gather[attrib_name] = gather[attrib_name][::-1]
    # then, flip the offset distances (this is a 3-ch array, if it is defined at all)
    if "offset_distances" in gather and gather["offset_distances"] is not None:
        assert gather["offset_distances"].shape == (n_traces, 3)
        # first ch = shot distance (OK to flip), last two ch = next/previous distance (need swap!)
        gather["offset_distances"] = gather["offset_distances"][::-1, [0, 2, 1]]
    # next, flip the samples array (this is a 2d array)
    assert gather["samples"].shape == (n_traces, n_samples)
    gather["samples"] = gather["samples"][::-1, :]
    # finally, flip the segmentation mask array (this is a 2d array, if it is defined at all)
    if "segmentation_mask" in gather and gather["segmentation_mask"] is not None:
        gather["segmentation_mask"] = gather["segmentation_mask"][::-1, :]


def generate_segmentation_mask(
    gather: GatherType,
    segm_class_count: int,
    segm_first_break_buffer: int = 0,
):
    """Generates the segmentation masks for a given gather with a particular class count."""
    assert isinstance(gather, dict)
    assert "first_break_labels" in gather and "samples" in gather
    fb_labels, samples = gather["first_break_labels"], gather["samples"]
    assert len(fb_labels.shape) == 1 and len(samples.shape) == 2
    assert fb_labels.max() < samples.shape[1]
    fb_mask = np.full_like(
        samples,
        fill_value=generic_consts.DONTCARE_SEGM_MASK_LABEL,
        dtype=np.int32,
    )
    for trace_idx, fb_label in enumerate(fb_labels):
        if fb_label > consts.BAD_FIRST_BREAK_PICK_INDEX:
            if segm_class_count in [1, 3]:
                min_sample_idx = max(fb_label - segm_first_break_buffer, 0)
                max_sample_idx = min(fb_label + 1 + segm_first_break_buffer, samples.shape[1])
                assert max_sample_idx - min_sample_idx > 0
                if segm_class_count == 1:
                    fb_mask[trace_idx, :min_sample_idx] = 0  # before first break = class idx 0
                    fb_mask[trace_idx, min_sample_idx:max_sample_idx] = 1  # first break = class idx 1
                    fb_mask[trace_idx, max_sample_idx:] = 0  # after first break = class idx 0
                else:  # if segm_class_count == 3:
                    fb_mask[trace_idx, :min_sample_idx] = 0  # before first break = class idx 0
                    fb_mask[trace_idx, min_sample_idx:max_sample_idx] = 2  # first break = class idx 2
                    fb_mask[trace_idx, max_sample_idx:] = 1  # after first break = class idx 1
            elif segm_class_count == 2:
                fb_mask[trace_idx, :fb_label] = 0  # before first break = class idx 0
                fb_mask[trace_idx, fb_label:] = 1  # at/after first break = class idx 1
    gather["segmentation_mask"] = fb_mask


def generate_prior_mask(
    gather: GatherType,
    prior_velocity_range: typing.Tuple[float, float],  # (min, max) in m/s
    prior_offset_range: typing.Tuple[float, float],  # (min, max) in milliseconds
):
    """Generates the first break prior channel (masks) for a given gather with particular ranges."""
    assert isinstance(gather, dict)
    assert len(prior_velocity_range) == 2 and prior_velocity_range[1] > prior_velocity_range[0]
    assert len(prior_offset_range) == 2 and prior_offset_range[1] > prior_offset_range[0]
    assert prior_velocity_range[0] > 0 and prior_offset_range[0] >= 0
    assert "trace_count" in gather and "sample_count" in gather and "sample_rate_ms" in gather
    trace_count, sample_count = gather["trace_count"], gather["sample_count"]
    sample_rate_ms = gather["sample_rate_ms"]
    assert "offset_distances" in gather, "need to generate offset distances for prior mask!"
    offset_distances = gather["offset_distances"][:, 0]  # keep shot-to-rec offsets only
    expected_range_sec = np.stack((
        offset_distances / prior_velocity_range[1] + prior_offset_range[0] / 1000,
        offset_distances / prior_velocity_range[0] + prior_offset_range[1] / 1000,
    ), axis=1)
    expected_range_idxs = np.round(expected_range_sec / (sample_rate_ms / 1000)).astype(np.int32)
    assert all([idxs[0] <= idxs[1] for idxs in expected_range_idxs])
    fb_mask = np.zeros((trace_count, sample_count), dtype=np.float32)
    for trace_idx, sample_range_idxs in enumerate(expected_range_idxs):
        fb_mask[trace_idx, sample_range_idxs[0]:sample_range_idxs[1]] = 1
    gather["first_break_prior"] = fb_mask
