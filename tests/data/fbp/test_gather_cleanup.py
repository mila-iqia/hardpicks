import mock
import numpy as np
import pytest
import yaml

import hardpicks.data.fbp.gather_parser as gather_parser
import hardpicks.data.fbp.constants as consts
import hardpicks.data.fbp.gather_cleaner as gather_cleaner

OFFSET_DIST_MAX_ERROR = 5  # 5m; all distances are roughly rounded, we're generous here
# ... note: on a receiver line, two receivers are rarely closer than 15m apart, so 5m is OK


@pytest.mark.slow
@pytest.mark.parametrize("real_site_name", ["Sudbury", "Halfmile"])
def test_gather_distances_computed_vs_real(real_dataset):
    if real_dataset is None:
        pytest.skip("missing hdf5 file for real-data test. Skipping test.")
    max_iter_count = 100
    for sample_idx in range(len(real_dataset)):
        sample = real_dataset[sample_idx]
        # by default, the line-shot-gather loaded does not parse the "OFFSET" field at all
        # ...we'll do it here manually just to make sure that the values are all similar
        gather_trace_ids = sample["gather_trace_ids"]
        assert real_dataset._worker_h5fd is not None
        offset_dataset = real_dataset._worker_h5fd["TRACE_DATA"]["DEFAULT"]["OFFSET"]
        orig_distances = offset_dataset[gather_trace_ids].flatten()
        assert sample["offset_distances"] is not None
        assert not (sample["offset_distances"][:, 0] < 0).any()
        computed_distances = sample["offset_distances"][
            :, 0
        ].flatten()  # 0th channel = shot-rec dists
        assert len(computed_distances) == len(orig_distances)
        distances_diffs = np.abs(orig_distances - computed_distances)
        assert np.isclose(distances_diffs, 0, atol=OFFSET_DIST_MAX_ERROR).all()
        if sample_idx > max_iter_count:
            break


class FakeMiniDataset:
    def __init__(self):
        self.data = [
            {"dummy_attrib": 0},
            {"dummy_attrib": 1},
            {"dummy_attrib": 3},
        ]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _is_gather_valid(item):
        return True

    def __getitem__(self, item):
        return {**self.get_meta_gather(item), **self.data[item]}

    def get_meta_gather(self, item):
        return {
            "origin": "dummy site name",
            "gather_id": item,
            "bad_first_breaks_mask": np.zeros(1),
        }


@pytest.fixture()
def fake_mini_dataset():
    return FakeMiniDataset()


def test_gather_cleaner_dataset_wrapper(fake_mini_dataset):
    # we'll just make a dummy dataset with a compatible interface and see that indexing works OK
    with mock.patch.object(gather_parser, "ShotLineGatherDataset", new=FakeMiniDataset):
        wrapped_dataset = gather_cleaner.ShotLineGatherCleaner(
            dataset=fake_mini_dataset,
            # just turn off all the actual cleaning and check that the dataset is untouched
            auto_invalidate_outlier_picks=False,
            auto_fill_missing_picks=False,
            pick_blending_factor=0.0,
        )
        assert len(fake_mini_dataset) == len(wrapped_dataset)
        for sample_idx in range(len(fake_mini_dataset)):
            orig_sample = fake_mini_dataset[sample_idx]
            with mock.patch.object(wrapped_dataset, "_post_process_gather_in_place") as mfunc:
                wrapped_sample = wrapped_dataset[sample_idx]
                assert mfunc.call_count == 1
            for attr in orig_sample.keys():
                assert orig_sample[attr] == wrapped_sample[attr]
        with mock.patch.object(fake_mini_dataset, "_is_gather_valid", return_value=False):
            # in this case, no sample should make it through (all gathers are invalid)
            wrapped_dataset = gather_cleaner.ShotLineGatherCleaner(
                dataset=fake_mini_dataset,
                auto_invalidate_outlier_picks=False,
                auto_fill_missing_picks=False,
                pick_blending_factor=0.0,
            )
        assert len(wrapped_dataset) == 0


@pytest.fixture(scope="session")
def fake_outlier_sample():
    np.random.seed(1337)
    trace_count = 1000  # make the gather pretty wide...
    shot_closest_idx = np.random.randint(
        trace_count
    )  # this is where the dome tip will be
    offset_distances = (
        np.abs(np.arange(trace_count) - shot_closest_idx).astype(np.float32) + 10
    )
    offset_distances += np.random.normal(
        scale=1.0, size=(trace_count,)
    )  # make offsets noisy
    first_break_timestamps = (
        offset_distances * 0.01
    )  # velocity scale doesn't matter here
    first_break_timestamps += np.random.normal(
        scale=0.01, size=(trace_count,)
    )  # also make noisy

    # first, generate a bunch of already-invalid picks:
    bad_pick_mask = np.random.choice([False, True], size=(trace_count,), p=(0.9, 0.1))
    first_break_timestamps[bad_pick_mask] = consts.BAD_FIRST_BREAK_PICK_INDEX

    # next, make some outliers on purpose... (obvious ones just to make sure we catch them)
    outlier_gt_mask = np.random.choice(
        [False, True], size=(trace_count,), p=(0.95, 0.05)
    )
    outlier_gt_idxs = np.flatnonzero(outlier_gt_mask)
    for outlier_idx in outlier_gt_idxs:
        if not bad_pick_mask[outlier_idx]:
            first_break_timestamps[outlier_idx] += 50  # obvious!
    return offset_distances, bad_pick_mask, first_break_timestamps, outlier_gt_mask


def test_gather_outlier_fbp_detection(fake_outlier_sample):
    # here, we'll bypass the creation of a real dataset and just call the flagging function directly
    offset_distances, bad_pick_mask, first_break_timestamps, outlier_gt_mask = (
        fake_outlier_sample
    )
    # assemble the actual sample now, and put it through the flagging function
    sample = dict(
        offset_distances=offset_distances.copy(),
        bad_first_breaks_mask=bad_pick_mask.copy(),
        first_break_timestamps=first_break_timestamps.copy(),
        first_break_labels=None,  # should not be actually needed for the outlier detection
    )
    with mock.patch.object(gather_parser, "ShotLineGatherDataset", new=list):
        wrapped_dataset = gather_cleaner.ShotLineGatherCleaner(
            dataset=[],  # just bypass this here, it shouldn't matter
            auto_invalidate_outlier_picks=True,
            outlier_detection_strategy="median-velocity-diff",
            outlier_detection_filter_size=15,
            outlier_detection_threshold=10.0,
            # the other cleaning ops should be turned off
            auto_fill_missing_picks=False,
            pick_blending_factor=0.0,
        )
    wrapped_dataset._find_and_flag_outliers(sample)

    # the provided arrays should not have been modified!
    np.testing.assert_array_equal(sample["offset_distances"], offset_distances)
    np.testing.assert_array_equal(sample["bad_first_breaks_mask"], bad_pick_mask)
    np.testing.assert_array_equal(
        sample["first_break_timestamps"], first_break_timestamps
    )

    assert "outlier_first_breaks_mask" in sample
    outlier_first_breaks_mask = sample["outlier_first_breaks_mask"]

    for orig_bad_pick_idx in np.flatnonzero(bad_pick_mask):
        # all originally tagged bad picks should still be tagged as outliers
        assert outlier_first_breaks_mask[orig_bad_pick_idx]

    # note: with the noise, there might be more detections than real outliers, but that's OK
    for outlier_pick_idx in np.flatnonzero(outlier_gt_mask):
        # ... we just need a perfect recall
        assert outlier_first_breaks_mask[outlier_pick_idx]


@pytest.mark.slow
@pytest.mark.parametrize("real_site_name", ["Halfmile"])
def test_gather_clean_and_fill_real(real_site_info, real_dataset):
    if real_dataset is None:
        pytest.skip("missing hdf5 file for real-data test. Skipping test.")

    wrapped_dataset = gather_cleaner.ShotLineGatherCleaner(
        dataset=real_dataset,
        auto_invalidate_outlier_picks=True,
        outlier_detection_strategy="median-velocity-diff",
        outlier_detection_filter_size=15,
        outlier_detection_threshold=0.1,
        # the next two params are fill-related
        auto_fill_missing_picks=True,
        pick_fill_strategy="full-linear-fit",
        # the blending should be turned off
        pick_blending_factor=0.0,
    )
    max_iter_count = 10
    found_some_outliers, found_some_filled_vals = False, False
    for sample_idx in range(len(wrapped_dataset)):
        sample = wrapped_dataset[sample_idx]
        outliers_mask = sample["outlier_first_breaks_mask"]
        bad_picks_mask = sample["bad_first_breaks_mask"]
        filled_mask = sample["filled_first_breaks_mask"]
        assert outliers_mask.shape == bad_picks_mask.shape
        valid_idxs = np.flatnonzero(~bad_picks_mask)
        invalid_idxs = np.flatnonzero(bad_picks_mask)
        # all 'bad' picks should automatically be counted as outliers
        assert outliers_mask[invalid_idxs].all()
        found_some_outliers = found_some_outliers or outliers_mask[valid_idxs].any()
        # all first breaks in a gather cannot be outliers; there has to be some good values
        assert not outliers_mask.all()
        found_some_filled_vals = found_some_filled_vals or filled_mask.any()
        # 'filled' values should only be those that are 'bad' or 'outliers'!
        assert not np.logical_and(filled_mask, np.logical_not(outliers_mask)).any()
        if sample_idx > max_iter_count:
            break
    # on halfmile lake, with a filter size of 15 and a threshold of 0.1, we should find some outliers
    assert found_some_outliers and found_some_filled_vals


@pytest.fixture(scope="session")
def fake_tofill_sample():
    # this sample will not be noisy: we'll use it to test interpolation/extrapolation
    np.random.seed(42)
    trace_count = 1000  # make the gather pretty wide...
    shot_closest_idx = np.random.randint(
        trace_count
    )  # this is where the dome tip will be
    offset_distances = (
        np.abs(np.arange(trace_count) - shot_closest_idx).astype(np.float32) + 10
    )
    first_break_timestamps = offset_distances * 0.01  # velocity scale doesn't matter
    first_break_timestamps_gt = first_break_timestamps.copy()

    # first, generate a bunch of already-invalid picks (randomly located):
    bad_pick_mask = np.random.choice([False, True], size=(trace_count,), p=(0.9, 0.1))
    first_break_timestamps[bad_pick_mask] = consts.BAD_FIRST_BREAK_PICK_INDEX

    # now, make a couple of big empty sections to test long-range interpolation and extrapolation
    bad_pick_mask[0:100] = True
    first_break_timestamps[0:100] = consts.BAD_FIRST_BREAK_PICK_INDEX
    bad_pick_mask[500:550] = True
    first_break_timestamps[500:550] = consts.BAD_FIRST_BREAK_PICK_INDEX

    return (
        offset_distances,
        bad_pick_mask,
        first_break_timestamps,
        first_break_timestamps_gt,
    )


def test_gather_fbp_fill_linear(fake_tofill_sample):
    # here, we'll bypass the creation of a real dataset and just call the interp function directly
    offset_distances, bad_pick_mask, first_break_timestamps, first_break_timestamps_gt = (
        fake_tofill_sample
    )
    # assemble the actual sample now, and put it through the filling function
    samp_num = 10000
    max_fb_timestamp = 100.0
    samp_rate = (max_fb_timestamp * 1000) // samp_num
    dataset = mock.Mock(
        samp_num=samp_num, max_fb_timestamp=max_fb_timestamp, samp_rate=samp_rate
    )

    first_break_labels = np.full(first_break_timestamps.shape, -1, dtype=np.int32)
    for idx in np.flatnonzero(~bad_pick_mask):
        first_break_labels[idx] = int(
            round((first_break_timestamps[idx] / max_fb_timestamp) * samp_num)
        )
    sample = dict(
        offset_distances=offset_distances.copy(),
        bad_first_breaks_mask=bad_pick_mask.copy(),
        first_break_timestamps=first_break_timestamps.copy(),
        first_break_labels=first_break_labels.copy(),
    )

    with mock.patch.object(gather_cleaner.ShotLineGatherCleaner, "_get_valid_gather_ids"), \
            mock.patch.object(gather_parser, "ShotLineGatherDataset", new=mock.Mock):
        wrapped_dataset = gather_cleaner.ShotLineGatherCleaner(
            dataset=dataset,  # just bypass this here, it shouldn't matter
            # just for fun, we'll run the outlier detector as well, but it should be blind
            auto_invalidate_outlier_picks=True,
            outlier_detection_strategy="median-velocity-diff",
            outlier_detection_filter_size=15,
            outlier_detection_threshold=0.01,
            # the next two params are fill-related
            auto_fill_missing_picks=True,
            pick_fill_strategy="full-linear-fit",
            # the blending should be turned off
            pick_blending_factor=0.0,
        )

    wrapped_dataset._find_and_flag_outliers(sample)
    # should not have detected new outliers
    np.testing.assert_array_equal(sample["outlier_first_breaks_mask"], bad_pick_mask)
    wrapped_dataset._fill_missing_picks(sample)

    # the first two provided arrays should not have been modified!
    np.testing.assert_array_equal(sample["offset_distances"], offset_distances)
    np.testing.assert_array_equal(sample["bad_first_breaks_mask"], bad_pick_mask)
    # on the other hand, first break timestamps should have been modified...
    assert not np.array_equal(sample["first_break_timestamps"], first_break_timestamps)
    # ... but only on idxs that correspond to bad picks!
    np.testing.assert_array_equal(
        sample["first_break_timestamps"][~bad_pick_mask],
        first_break_timestamps[~bad_pick_mask],
    )

    assert "filled_first_breaks_mask" in sample
    filled_first_breaks_mask = sample["filled_first_breaks_mask"]
    # all the 'bad' picks should have been filled in
    np.testing.assert_array_equal(
        np.flatnonzero(filled_first_breaks_mask), np.flatnonzero(bad_pick_mask)
    )

    # finally, check that the interp version is close to the gt
    first_break_timestamps = sample["first_break_timestamps"]
    filled_idxs = np.flatnonzero(filled_first_breaks_mask)
    for pick_idx in filled_idxs:
        assert np.isclose(
            first_break_timestamps_gt[pick_idx], first_break_timestamps[pick_idx]
        )


@pytest.fixture()
def fake_bad_gather_ids():

    bad_gathers = dict()

    bad_gathers["siteA"] = {123: {'ReceiverId': 111, 'ShotId': 3321},
                            456: {'ReceiverId': 441, 'ShotId': 341},
                            }

    bad_gathers["siteB"] = {321: {'ReceiverId': 12, 'ShotId': 21},
                            999: {'ReceiverId': 41, 'ShotId': 31},
                            }

    bad_gathers["siteC"] = {0: {'ReceiverId': 1, 'ShotId': 2},
                            }

    return bad_gathers


@pytest.fixture()
def rejected_gather_yaml_path(fake_bad_gather_ids, tmpdir):
    rejected_gather_yaml_path = tmpdir.join("test_bad_gathers.yaml")
    with open(rejected_gather_yaml_path, "w") as fd:
        yaml.dump(fake_bad_gather_ids, fd)
    return rejected_gather_yaml_path


def test_get_rejected_gather_map(fake_bad_gather_ids, rejected_gather_yaml_path):

    computed_bad_ids = gather_cleaner.ShotLineGatherCleaner._get_rejected_gather_map(rejected_gather_yaml_path)

    assert set(computed_bad_ids.keys()) == set(fake_bad_gather_ids.keys())

    for site_name in fake_bad_gather_ids.keys():
        computed_ids = computed_bad_ids[site_name]
        expected_ids = list(np.sort(list(fake_bad_gather_ids[site_name].keys())))

        np.testing.assert_array_equal(computed_ids, expected_ids)
