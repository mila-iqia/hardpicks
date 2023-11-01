import copy

import deepdiff
import numpy as np
import pytest
import hardpicks.data.fbp.collate as collate
import hardpicks.data.fbp.constants as consts
import hardpicks.data.fbp.gather_transforms as transforms
import hardpicks.models.constants as generic_consts


def generate_fake_gather(
    trace_count: int,
    sample_count: int,
):
    assert trace_count > 0 and sample_count > 0
    first_break_labels = np.random.randint(0, sample_count, (trace_count,))
    offset_dists = np.random.rand(trace_count, 3)
    offset_dists[-1, 1], offset_dists[0, 2] = 0., 0.
    return {
        "sample_rate_ms": 1.,
        "rec_ids": np.arange(trace_count),
        "trace_count": trace_count,
        "sample_count": sample_count,
        "samples": np.random.rand(trace_count, sample_count),
        "first_break_labels": first_break_labels,
        "first_break_timestamps": first_break_labels,
        "bad_first_breaks_mask": np.random.choice([True, False], (trace_count,)),
        "offset_distances": offset_dists,
    }


def test_drop_trace_idxs():
    fake_gather = generate_fake_gather(100, 200)
    orig_gather = copy.deepcopy(fake_gather)
    transforms._drop_trace_idxs(fake_gather, [10, 99])
    assert fake_gather["trace_count"] == 98
    orig_offsets, new_offsets = orig_gather["offset_distances"], fake_gather["offset_distances"]
    assert len(new_offsets) == 98
    assert np.isclose(orig_offsets[9:11, 1].sum(), new_offsets[9, 1])
    assert np.isclose(orig_offsets[10:12, 2].sum(), new_offsets[10, 2])
    assert new_offsets[97, 1] == 0.
    assert new_offsets[97, 2] == orig_offsets[98, 2]
    transforms._drop_trace_idxs(fake_gather, [0])
    assert fake_gather["trace_count"] == 97
    assert np.isclose(orig_offsets[1, 1], fake_gather["offset_distances"][0, 1])
    assert fake_gather["offset_distances"][0, 2] == 0.


def test_drop_traces_no_bad_picks():
    fake_gather = generate_fake_gather(100, 200)
    fake_gather["bad_first_breaks_mask"] = np.zeros_like(fake_gather["bad_first_breaks_mask"])
    orig_gather = copy.deepcopy(fake_gather)
    transforms.drop_traces(fake_gather, 10, drop_bad_picks_first=True, drop_edges_next=True)
    assert fake_gather["trace_count"] == 90 and len(fake_gather["rec_ids"]) == 90
    assert np.array_equal(fake_gather["rec_ids"], np.arange(5, 95))
    transforms.drop_traces(orig_gather, 10, drop_bad_picks_first=True, drop_edges_next=False)
    assert orig_gather["trace_count"] == 90 and len(orig_gather["rec_ids"]) == 90
    assert not np.array_equal(orig_gather["rec_ids"], np.arange(5, 95))


def test_drop_traces_with_bad_pick():
    fake_gather = generate_fake_gather(100, 200)
    fake_gather["bad_first_breaks_mask"] = np.zeros_like(fake_gather["bad_first_breaks_mask"])
    fake_gather["bad_first_breaks_mask"][3], fake_gather["bad_first_breaks_mask"][22] = 1, 1
    transforms.drop_traces(fake_gather, 10, drop_bad_picks_first=True, drop_edges_next=True)
    assert fake_gather["trace_count"] == 90 and len(fake_gather["rec_ids"]) == 90
    assert np.array_equal(
        fake_gather["rec_ids"],
        np.concatenate([list(range(5, 22)), list(range(23, 96))])
    )


@pytest.mark.parametrize("prepad,postpad", [(0, 0), (0, 1), (3, 0), (4, 7)])
def test_pad_traces(
    prepad: int,
    postpad: int,
):
    orig_trace_count, orig_sample_count = 50, 100
    gather = generate_fake_gather(trace_count=orig_trace_count, sample_count=orig_sample_count)
    orig_gather = copy.deepcopy(gather)
    transforms.pad_traces(gather, prepad_trace_count=prepad, postpad_trace_count=postpad)
    expected_trace_count = orig_trace_count + prepad + postpad
    assert gather["trace_count"] == expected_trace_count
    assert gather["sample_count"] == orig_sample_count
    assert gather["samples"].shape == (expected_trace_count, orig_sample_count)
    assert gather["offset_distances"].shape == (expected_trace_count, 3)
    assert len(gather["first_break_labels"]) == expected_trace_count
    assert len(gather["bad_first_breaks_mask"]) == expected_trace_count
    for attrib_name, attrib_pad_val in collate.get_fields_to_pad():
        if attrib_name in gather:
            assert (gather[attrib_name][:prepad, ...] == attrib_pad_val).all()
            if postpad > 0:
                assert (gather[attrib_name][-postpad:, ...] == attrib_pad_val).all()
            assert np.array_equal(
                gather[attrib_name][prepad:expected_trace_count - postpad],
                orig_gather[attrib_name],
            )


@pytest.mark.parametrize("kill_prob", [0.01, 0.1, 0.5, 0.99])
def test_kill_traces(
    kill_prob: float,
):
    orig_trace_count, orig_sample_count = 100000, 10
    gather = generate_fake_gather(trace_count=orig_trace_count, sample_count=orig_sample_count)
    orig_dead_mask = (np.abs(gather["samples"]) <= consts.DEAD_TRACE_AMPLITUDE_TOLERANCE).all(axis=1)
    transforms.kill_traces(gather, kill_prob)
    new_dead_mask = (np.abs(gather["samples"]) <= consts.DEAD_TRACE_AMPLITUDE_TOLERANCE).all(axis=1)
    assert np.count_nonzero(orig_dead_mask) <= np.count_nonzero(new_dead_mask)
    new_dead_count = np.count_nonzero(new_dead_mask) - np.count_nonzero(orig_dead_mask)
    expected_dead_count = int(round(orig_trace_count * kill_prob))
    assert_buffer_size = int(round(orig_trace_count * 0.005))  # .5% should be generous enough?
    assert np.abs(new_dead_count - expected_dead_count) < assert_buffer_size


def test_add_noise_patch():
    orig_trace_count, orig_sample_count = 256, 1024
    for _ in range(1000):
        gather = generate_fake_gather(trace_count=orig_trace_count, sample_count=orig_sample_count)
        # to make the noise obvious, we'll just make the sample array a flat one
        gather["samples"] = np.ones((gather["trace_count"], gather["sample_count"]))
        orig_gather = copy.deepcopy(gather)
        transforms.add_noise_patch(
            gather=gather,
            prob=1.0,
        )
        assert not deepdiff.DeepDiff(
            {key: val for key, val in gather.items() if key != "samples"},
            {key: val for key, val in orig_gather.items() if key != "samples"},
        )
        assert not np.array_equal(orig_gather["samples"], gather["samples"])
        orig_max_ampl = np.abs(orig_gather["samples"]).max()
        assert np.abs(gather["samples"]).max() < 2 * orig_max_ampl


@pytest.mark.parametrize("target_crop_size", [25, 50, 100, 150])
def test_crop_samples(
    target_crop_size: int,
):
    orig_trace_count, orig_sample_count = 125, 100
    gather = generate_fake_gather(trace_count=orig_trace_count, sample_count=orig_sample_count)
    orig_gather = copy.deepcopy(gather)
    transforms.crop_samples(gather, max_samples=target_crop_size)
    expected_sample_count = min(orig_sample_count, target_crop_size)
    assert gather["trace_count"] == orig_trace_count
    assert gather["sample_count"] == expected_sample_count
    assert gather["samples"].shape == (orig_trace_count, expected_sample_count)
    assert np.array_equal(gather["samples"], orig_gather["samples"][:, :target_crop_size])
    new_oob_fb_mask = orig_gather["first_break_labels"] >= target_crop_size
    assert gather["bad_first_breaks_mask"][new_oob_fb_mask].all()
    assert (gather["first_break_labels"][new_oob_fb_mask] == consts.BAD_FIRST_BREAK_PICK_INDEX).all()


@pytest.mark.parametrize("target_sample_count", [25, 50, 75, 100, 125, 150])
def test_resample(
    target_sample_count: int,
):
    orig_trace_count, orig_sample_count = 125, 100
    gather = generate_fake_gather(trace_count=orig_trace_count, sample_count=orig_sample_count)
    orig_gather = copy.deepcopy(gather)
    assert not (orig_gather["first_break_labels"] >= orig_sample_count).any()
    transforms.resample(gather, target_sample_count=target_sample_count)
    assert gather["trace_count"] == orig_trace_count
    assert gather["sample_count"] == target_sample_count
    assert gather["samples"].shape == (orig_trace_count, target_sample_count)
    if target_sample_count != orig_sample_count:
        assert not np.isclose(gather["sample_rate_ms"], orig_gather["sample_rate_ms"])
        assert not (gather["first_break_labels"] >= target_sample_count).any()
        transforms.resample(gather, target_sample_count=orig_sample_count)
        assert gather["sample_count"] == orig_sample_count
        assert gather["samples"].shape == (orig_trace_count, orig_sample_count)
        assert gather["sample_rate_ms"] == orig_gather["sample_rate_ms"]
        assert np.isclose(gather["sample_rate_ms"], orig_gather["sample_rate_ms"])
        assert not (gather["first_break_labels"] >= orig_sample_count).any()
        if target_sample_count > orig_sample_count:
            assert np.allclose(gather["samples"], orig_gather["samples"])


def test_flip():
    orig_trace_count, orig_sample_count = 125, 100
    gather = generate_fake_gather(trace_count=orig_trace_count, sample_count=orig_sample_count)
    orig_gather = copy.deepcopy(gather)
    transforms.flip(gather)
    assert gather["trace_count"] == orig_trace_count
    assert gather["sample_count"] == orig_sample_count
    assert gather["samples"].shape == (orig_trace_count, orig_sample_count)
    for attrib_name, attrib_pad_val in collate.get_fields_to_pad():
        if attrib_name in gather:
            assert not np.array_equal(gather[attrib_name], orig_gather[attrib_name])
    assert np.array_equal(gather["samples"][::-1, :], orig_gather["samples"])
    transforms.flip(gather)
    for attrib_name, attrib_pad_val in collate.get_fields_to_pad():
        if attrib_name in gather:
            assert np.array_equal(gather[attrib_name], orig_gather[attrib_name])


@pytest.mark.parametrize("segm_class_count", [1, 2, 3])
def test_generate_segmentation_mask_no_buffer(segm_class_count: int):
    orig_trace_count, orig_sample_count = 50, 100
    gather = generate_fake_gather(trace_count=orig_trace_count, sample_count=orig_sample_count)
    gather["bad_first_breaks_mask"] = gather["first_break_labels"] <= consts.BAD_FIRST_BREAK_PICK_INDEX
    assert "segmentation_mask" not in gather
    transforms.generate_segmentation_mask(gather, segm_class_count=segm_class_count)
    assert "segmentation_mask" in gather
    segm_mask = gather["segmentation_mask"]
    bad_trace_idxs = np.where(gather["bad_first_breaks_mask"])
    assert (segm_mask[bad_trace_idxs] == generic_consts.DONTCARE_SEGM_MASK_LABEL).all()
    good_trace_idxs = np.where(np.logical_not(gather["bad_first_breaks_mask"]))
    assert not (segm_mask[good_trace_idxs] == generic_consts.DONTCARE_SEGM_MASK_LABEL).any()
    real_class_count = max(segm_class_count, 2)
    assert (segm_mask[good_trace_idxs] < real_class_count).all()


@pytest.mark.parametrize("buffer", [1, 3, 5])
def test_generate_segmentation_mask_buffer(buffer: int):
    orig_trace_count, orig_sample_count = 50, 100
    gather = generate_fake_gather(trace_count=orig_trace_count, sample_count=orig_sample_count)
    gather["bad_first_breaks_mask"] = gather["first_break_labels"] <= consts.BAD_FIRST_BREAK_PICK_INDEX
    transforms.generate_segmentation_mask(gather, segm_class_count=3, segm_first_break_buffer=buffer)
    segm_mask = gather["segmentation_mask"]
    good_trace_idxs = np.where(np.logical_not(gather["bad_first_breaks_mask"]))[0]
    for trace_idx in good_trace_idxs:
        expected_fb_idx = gather["first_break_labels"][trace_idx]
        assert segm_mask[trace_idx, expected_fb_idx] == 2  # 2 == first break class in 3-class setup
        min_buffer_idx = max(expected_fb_idx - buffer, 0)
        max_buffer_idx = min(expected_fb_idx + buffer + 1, orig_sample_count - 1)
        assert (segm_mask[trace_idx, min_buffer_idx:max_buffer_idx] == 2).all()


def test_generate_prior_mask():
    for _ in range(100):
        trace_count, sample_count = 128, 2048
        sample_rate_ms = np.random.choice([1, 2, 4])
        offset_distances = np.random.rand(trace_count, 3) * 5000
        gather = {
            "trace_count": trace_count,
            "sample_count": sample_count,
            "sample_rate_ms": sample_rate_ms,
            "offset_distances": offset_distances.copy(),
        }
        prior_velocity_range_min, prior_velocity_range_max = 4000, 7000
        prior_offset_range_min, prior_offset_range_max = 10, 50
        transforms.generate_prior_mask(
            gather,
            (prior_velocity_range_min, prior_velocity_range_max),
            (prior_offset_range_min, prior_offset_range_max),
        )
        assert gather["trace_count"] == trace_count
        assert gather["sample_count"] == sample_count
        assert gather["sample_rate_ms"] == sample_rate_ms
        assert np.array_equal(gather["offset_distances"], offset_distances)
        assert "first_break_prior" in gather
        assert isinstance(gather["first_break_prior"], np.ndarray)
        for row_idx, prior_row in enumerate(gather["first_break_prior"]):
            min_mask_idx = np.argmax(prior_row)
            min_mask_tstamp = (min_mask_idx * sample_rate_ms - prior_offset_range_min) / 1000
            min_expected_tstamp = offset_distances[row_idx, 0] / prior_velocity_range_max
            assert np.isclose(min_expected_tstamp, min_mask_tstamp, atol=sample_rate_ms / 1000)
            max_mask_idx = np.argmin(prior_row[min_mask_idx:]) + min_mask_idx
            max_mask_tstamp = (max_mask_idx * sample_rate_ms - prior_offset_range_max) / 1000
            max_expected_tstamp = offset_distances[row_idx, 0] / prior_velocity_range_min
            assert np.isclose(max_expected_tstamp, max_mask_tstamp, atol=sample_rate_ms / 1000)
            assert min_mask_tstamp <= max_mask_tstamp
