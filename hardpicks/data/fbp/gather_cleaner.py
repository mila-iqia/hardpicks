"""Trace gather cleaning module.

The amount of 'cleaning' done here will be limited to simple in-trace corrections that
should not affect runtime performance much. This is in contrast with the 'correction'
module which rewrites new HDF5 files while fixing their internal issues.
"""

import logging
import os
import typing

import numpy as np
import scipy.interpolate
import yaml

import hardpicks.data.fbp.constants as consts
import hardpicks.data.fbp.gather_parser as gather_parser
import hardpicks.data.fbp.line_gather_cleaning_utils as line_clean_utils
from hardpicks.data.utils import get_value_or_default as vdefault

logger = logging.getLogger(__name__)


class ShotLineGatherCleaner(gather_parser.ShotLineGatherDatasetBase):
    """Wraps a shot (line) gather dataset parser object to clean its content.

    This object will behave exactly the same way as the gather dataset parser in the eyes of a
    data loader, but it will also filter/clean/fill the raw data that's read from the HDF5 files.

    It currently has three jobs: 1) to identify and flag outliers in the first break picks based on
    the distribution of expected first arrivals, 2) interpolate/extrapolate first break picks in
    regions where they are invalid, and 3) adjust the position of existing picks based on the
    interpolation of nearby picks. Each of these features can be individually deactivated based on
    the arguments given to the constructor.

    Two outlier detection strategies are currently implemented:
    1 - outlier_detection_strategy = "median-velocity-diff"

    2 - outlier_detection_strategy = "global-linear-fit-error-quantile"

        This strategy performs a linear fit on all traces, relating source-receiver distance (d)
        with first break pick times (t), of the form:
            d = v0 *( t- t0 )

        where v0 is the effective velocity and t0 the effective time offset.
        The absolute error with respect to this fit is then computed,
            e = | d - v0*(t-t0)|            (d, t real data, v0, t0 fitted parameters).

        The input parameter outlier_detection_threshold is used as the quantile above which an error
        is deemed "suspicious". As soon as a trace has an absolute error above this quantile, the
        whole line gather (ie, all traces in the line gather) is deemed suspicious.

    Attributes:
        dataset: reference to the dataset that provides the raw data to clean.
        auto_invalidate_outlier_picks: toggles whether outliers should be identified and overwritten
            in the loaded gather first break arrays.
        outlier_detection_strategy: defines the outlier detection strategy to use.
        outlier_detection_filter_size: defines the size of the outlier detection filter to use.
        outlier_detection_threshold: defines the sensitivity threshold of the outlier detection
            strategy. See the ``outlier_detection_strategies`` class attribute for more info.
        auto_fill_missing_picks: toggles whether missing/invalid/outlier picks should be replaced by
            interpolation or extrapolation estimates.
        pick_fill_strategy: defines the pick replacement strategy.  See the ``pick_fill_strategies``
            class attribute for more info.
        pick_fill_max_dist: defines the maximum distance within valid picks that new picks will be
            filled with estimations. Only useful with one of the `nearby` fill strategies.
        pick_blending_factor: Defines the blending factor that may be used to fuse pick
            estimations (obtained via interpolation/extrapolation) with real pick values. Using
            `0.0` will turn off the blending and keep original picks intact (if they were valid).

    """

    outlier_detection_strategies = [  # TODO add more strategies here if needed
        "median-velocity-diff",  # threshold the median velocity difference for each point
        "global-linear-fit-error-quantile",  # remove line gathers when it contains
        # a trace with an error in large quantile
        # TODO "linear-velocity-fit-dist",  # threshold the linear velocity model fit difference for each point
    ]

    pick_fill_strategies = [  # TODO add more strategies here if needed
        "full-linear-fit",  # fill all missing values with a linear distance-to-timestamp fit
        # TODO "nearby-linear-fit",  # fill all missing values within a maximum distance of a valid pick
    ]

    # note: all the fields listed below will need to be padded in the collate function
    variable_length_fields = [  # each tuple is the field name + the default fill value
        ("outlier_first_breaks_mask", True),
        ("filled_first_breaks_mask", False),
    ]

    def __init__(
        self,
        dataset: gather_parser.ShotLineGatherDataset,  # note: cannot be "base" interface, we need full!
        # these default to `None` so that config files can leave them empty and get real defaults below
        auto_invalidate_outlier_picks: typing.Optional[bool] = None,
        outlier_detection_strategy: typing.Optional[typing.AnyStr] = None,
        outlier_detection_filter_size: typing.Optional[int] = None,
        outlier_detection_threshold: typing.Optional[float] = None,
        auto_fill_missing_picks: typing.Optional[bool] = None,
        pick_fill_strategy: typing.Optional[typing.AnyStr] = None,
        pick_fill_max_dist: typing.Optional[int] = None,
        pick_blending_factor: typing.Optional[float] = None,
        rejected_gather_yaml_path: typing.Optional[typing.AnyStr] = None,
    ):
        """Validates and stores all the cleaning hyperparameters and a reference to the wrapped dataset."""
        assert isinstance(dataset, gather_parser.ShotLineGatherDataset), \
            "dataset cleaner needs to wrap the actual real gather parser (not an already-wrapped obj)"
        self.dataset = dataset

        # default value = yes, we will detect outliers and flag them
        self.auto_invalidate_outlier_picks = vdefault(auto_invalidate_outlier_picks, False)

        # default outlier detection strategy = threshold the distance to median velocity in local windows
        self.outlier_detection_strategy = vdefault(outlier_detection_strategy, "median-velocity-diff")
        assert self.outlier_detection_strategy in self.outlier_detection_strategies, \
            f"invalid outlier detection strategy: {outlier_detection_strategy}"

        # default outlier filtering window size = 15 traces wide (Bhavya's choice)
        self.outlier_detection_filter_size = vdefault(outlier_detection_filter_size, 15)
        assert self.outlier_detection_filter_size > 0, "invalid outlier detection filter size"

        # default outlier detection sensitivity threshold = 0.7 (Bhavya's choice)
        self.outlier_detection_threshold = vdefault(outlier_detection_threshold, 0.7)
        assert self.outlier_detection_threshold > 0, "invalid outlier detection threshold"

        if self.outlier_detection_strategy == "global-linear-fit-error-quantile":
            # prepare global quantities that are needed for this strategy
            assert 0.0 <= self.outlier_detection_threshold <= 1.0, (
                "The variable outlier_detection_threshold should be between 0 and 1: "
                "it is used as the quantile above which an error is deemed suspicious"
            )
            self.global_fit_velocity, self.global_fit_time_offset, all_absolute_errors = \
                line_clean_utils.get_global_linear_fit_and_all_errors(self.dataset)
            self.absolute_error_threshold = np.quantile(
                all_absolute_errors, self.outlier_detection_threshold
            )

        # by default, we will NOT fill in the labels for traces that have missing/bad picks
        self.auto_fill_missing_picks = vdefault(auto_fill_missing_picks, False)

        # by default, we will fill using a linear fit between offset distances and pick times
        self.pick_fill_strategy = vdefault(pick_fill_strategy, "full-linear-fit")
        assert self.pick_fill_strategy in self.pick_fill_strategies, \
            f"invalid pick fill strategy: {pick_fill_strategy}"

        # by default, we use an arbitrary 10-trace window to fill with somewhat-valid labels
        self.pick_fill_max_dist = vdefault(pick_fill_max_dist, 10)  # TODO: tune this default!
        assert self.pick_fill_max_dist > 0, "invalid pick fill maximum distance"

        # by default, we do not apply any blending between new fits and old pick values
        self.pick_blending_factor = vdefault(pick_blending_factor, 0.0)
        assert 0 <= self.pick_blending_factor < 1, "invalid pick blending factor"

        self.rejected_gather_map = self._get_rejected_gather_map(rejected_gather_yaml_path)

        # finally, get the real number of gathers with at least one valid pick
        self.valid_gather_ids = self._get_valid_gather_ids()

    @staticmethod
    def _get_rejected_gather_map(rejected_gather_yaml_path):
        # next, prepare the map of gather IDs that will be automatically rejected without analysis
        if rejected_gather_yaml_path is None:
            return {}

        assert os.path.isfile(rejected_gather_yaml_path), f"file {rejected_gather_yaml_path} does not exist."

        with open(rejected_gather_yaml_path, "r") as fd:
            input_rejected_gather_map = yaml.load(fd, Loader=yaml.FullLoader)

        assert isinstance(input_rejected_gather_map, dict), "invalid rejected gather map type"

        rejected_gather_map = {}
        for site_name in list(input_rejected_gather_map.keys()):
            rejected_gather_site_dict = input_rejected_gather_map[site_name]
            assert isinstance(rejected_gather_site_dict, dict)
            list_ids = list(rejected_gather_site_dict.keys())
            assert all([isinstance(i, int) for i in list_ids])
            rejected_gather_map[site_name] = np.sort(list_ids).tolist()
        return rejected_gather_map

    def _get_valid_gather_ids(self):
        """Returns the list of valid gather ids (i.e. all those who have at least one good pick)."""
        # TODO: we should probably also discard gathers that have less than X valid picks or less
        #       than Y traces in total here (just need to add new hyperparams & check in loop below)
        valid_gather_ids = []
        for orig_gather_id in range(len(self.dataset)):
            meta_gather = self.dataset.get_meta_gather(orig_gather_id)
            self._post_process_gather_in_place(meta_gather)
            if meta_gather["origin"] in self.rejected_gather_map \
                    and meta_gather["gather_id"] in self.rejected_gather_map[meta_gather["origin"]]:
                continue
            if self.auto_invalidate_outlier_picks:
                self._find_and_flag_outliers(meta_gather)
                if meta_gather["outlier_first_breaks_mask"].all():
                    continue
            else:
                if not self.dataset._is_gather_valid(orig_gather_id):
                    continue
            valid_gather_ids.append(orig_gather_id)
        return valid_gather_ids

    def __len__(self) -> int:
        """Returns the total number of gathers in the wrapped dataset."""
        return len(self.valid_gather_ids)

    def _get_original_gather_id(self, cleaned_gather_id):
        """Returns the original gather id for the underlying dataset."""
        assert (
            0 <= cleaned_gather_id < len(self.valid_gather_ids)
        ), "gather query index is out-of-bounds"
        orig_gather_id = self.valid_gather_ids[cleaned_gather_id]
        return orig_gather_id

    def _post_process_gather_in_place(self, gather):
        """Complete the gather depending on various strategies."""
        assert isinstance(gather, dict), "unexpected wrapped dataset gather type"
        gather["outlier_first_breaks_mask"] = gather["bad_first_breaks_mask"].copy()
        gather["filled_first_breaks_mask"] = np.full_like(gather["bad_first_breaks_mask"], False)
        if self.auto_invalidate_outlier_picks:
            self._find_and_flag_outliers(gather)
        if self.auto_fill_missing_picks:
            self._fill_missing_picks(gather)
        if self.pick_blending_factor > 0:  # pragma: no cover
            # self._blend_picks(gather)  TODO?
            raise NotImplementedError

    def __getitem__(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing all pertinent information for a particular gather."""
        orig_gather_id = self._get_original_gather_id(gather_id)
        gather = self.dataset[orig_gather_id]
        self._post_process_gather_in_place(gather)
        return gather

    def get_meta_gather(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing meta-information for a particular gather."""
        orig_gather_id = self._get_original_gather_id(gather_id)
        gather = self.dataset.get_meta_gather(orig_gather_id)
        self._post_process_gather_in_place(gather)
        return gather

    def _detect_outliers_based_on_global_linear_fit_error_quantile(
        self,
        bad_first_breaks_mask: np.ndarray,
        offset_distances: np.ndarray,
        first_break_timestamps: np.ndarray,
        valid_idxs: np.ndarray,
        invalid_idxs: np.ndarray,
    ) -> np.ndarray:
        """Implements the outlier detection strategy ``global-linear-fit-error-quantile``."""
        # NOTE: THIS FUNCTION WILL *NOT* MODIFY ANY OF THE ARGS IT IS PROVIDED.

        good_times = first_break_timestamps[valid_idxs].astype(np.float64)
        good_distances = offset_distances[valid_idxs].astype(np.float64)
        fitted_distances = line_clean_utils.get_fitted_distances(
            self.global_fit_velocity, self.global_fit_time_offset, good_times
        )

        absolute_errors = np.abs(good_distances - fitted_distances)

        line_gather_is_bad = np.any(absolute_errors >= self.absolute_error_threshold)

        outlier_first_breaks_mask = bad_first_breaks_mask.copy()
        outlier_first_breaks_mask[valid_idxs] = line_gather_is_bad

        return outlier_first_breaks_mask

    def _detect_outliers_based_on_median_velocity_dist(
        self,
        bad_first_breaks_mask: np.ndarray,
        offset_distances: np.ndarray,
        first_break_timestamps: np.ndarray,
        valid_idxs: np.ndarray,
        invalid_idxs: np.ndarray,
    ) -> np.ndarray:
        """Implements the first outlier detection strategy (``median-velocity-diff``)."""
        # NOTE: THIS FUNCTION WILL *NOT* MODIFY ANY OF THE ARGS IT IS PROVIDED.
        outlier_first_breaks_mask = bad_first_breaks_mask.copy()
        velocities = offset_distances / np.maximum(first_break_timestamps, 1e-6)
        velocities[invalid_idxs] = np.nan
        trace_count = len(offset_distances)
        median_filter_windows = np.full(
            (trace_count, self.outlier_detection_filter_size), np.nan, dtype=np.float32
        )
        for trace_idx in range(trace_count):
            if trace_idx in invalid_idxs:
                continue  # nothing more to do on already-invalid picks
            filter_win_start_idx = trace_idx - self.outlier_detection_filter_size // 2
            filter_win_end_idx = (
                filter_win_start_idx + self.outlier_detection_filter_size
            )
            filter_win_start_idx = max(filter_win_start_idx, 0)
            filter_win_end_idx = min(filter_win_end_idx, trace_count)
            filter_real_win_size = filter_win_end_idx - filter_win_start_idx
            median_filter_windows[trace_idx, :filter_real_win_size] = velocities[
                filter_win_start_idx:filter_win_end_idx
            ]
        assert (
            np.count_nonzero(~np.isnan(median_filter_windows[valid_idxs]), axis=1) > 0
        ).all()
        median_velocities = np.nanmedian(median_filter_windows[valid_idxs], axis=1)
        for valid_idx, median_velocity, curr_velocity in zip(
            valid_idxs, median_velocities, velocities[valid_idxs]
        ):
            velocity_diff = np.abs(median_velocity - curr_velocity)
            outlier_first_breaks_mask[valid_idx] = (
                velocity_diff > self.outlier_detection_threshold
            )
        return outlier_first_breaks_mask

    @staticmethod
    def _unpack_filter_vars(gather):
        """Unpacks variables used in filtering steps (outlier flagging + pick filling)."""
        # check the offset attrib since it's optionally computed, the rest we'll let fail on lookup
        assert (
            "offset_distances" in gather and gather["offset_distances"] is not None
        ), "missing required offset_distances attribute in gather data for outlier flagging"
        offset_distances = gather["offset_distances"]
        if offset_distances.ndim > 1:
            # if the offset distances array is not single-dimension, pick the 1st slice
            # (the other slices will be nearest-neighbor offsets that we don't need here)
            assert offset_distances.ndim == 2 and offset_distances.shape[1] == 3
            offset_distances = offset_distances[:, 0]
        first_break_labels = gather["first_break_labels"]
        first_break_timestamps = gather["first_break_timestamps"]
        to_fill_mask = gather["bad_first_breaks_mask"]
        if "outlier_first_breaks_mask" in gather:
            # if the outlier detection was applied, we'll use that mask instead of 'bad' picks only
            to_fill_mask = gather["outlier_first_breaks_mask"]
        valid_idxs = np.flatnonzero(~to_fill_mask)
        invalid_idxs = np.flatnonzero(to_fill_mask)
        return (
            offset_distances,
            first_break_labels,
            first_break_timestamps,
            to_fill_mask,
            valid_idxs,
            invalid_idxs,
        )

    def _find_and_flag_outliers(self, gather: typing.Dict):
        """This function will identify outliers and update the masks to flag them downstream."""
        offset_distances, first_break_labels, first_break_timestamps, bad_first_breaks_mask, \
            valid_idxs, invalid_idxs = self._unpack_filter_vars(gather)
        if len(invalid_idxs) == len(bad_first_breaks_mask):
            # all picks are invalid, nothing to do here
            return
        # we'll create a new mask and update it where appropriate based on the detection strategy
        # (note: by definition, 'bad' first breaks are automatically labeled as outliers as well)
        if self.outlier_detection_strategy == "median-velocity-diff":
            outlier_first_breaks_mask = self._detect_outliers_based_on_median_velocity_dist(
                bad_first_breaks_mask=bad_first_breaks_mask,
                offset_distances=offset_distances,
                first_break_timestamps=first_break_timestamps,
                valid_idxs=valid_idxs,
                invalid_idxs=invalid_idxs,
            )
        elif self.outlier_detection_strategy == "global-linear-fit-error-quantile":
            outlier_first_breaks_mask = self._detect_outliers_based_on_global_linear_fit_error_quantile(
                bad_first_breaks_mask=bad_first_breaks_mask,
                offset_distances=offset_distances,
                first_break_timestamps=first_break_timestamps,
                valid_idxs=valid_idxs,
                invalid_idxs=invalid_idxs,
            )
        else:  # pragma: no cover
            raise NotImplementedError
        # once we've got our newly filled outlier mask, we just add it to the gather dict
        gather["outlier_first_breaks_mask"] = outlier_first_breaks_mask

    @staticmethod
    def _compute_missing_picks_with_linear_fit(
        offset_distances: np.ndarray,
        first_break_timestamps: np.ndarray,
        valid_idxs: np.ndarray,
        invalid_idxs: np.ndarray,
        max_fb_timestamp: float,
        min_fb_timestamp: float,
    ) -> typing.Dict[int, float]:  # missing-index-to-computed-timestamp map
        """Implements the first fb timestamp filling strategy (``full-linear-fit``)."""
        # NOTE: THIS FUNCTION WILL *NOT* MODIFY ANY OF THE ARGS IT IS PROVIDED.
        # we need to sort the values, average Y's for duplicate X's, and send the result to scipy
        sorted_idxs = np.argsort(offset_distances[valid_idxs])
        x_array, y_array = [None], [[]]
        for idx in sorted_idxs:
            x = offset_distances[valid_idxs][idx]
            y = first_break_timestamps[valid_idxs][idx]
            if x != x_array[-1]:
                x_array.append(x)
                y_array.append([])
            y_array[-1].append(y)
        # need at least two points to do a linear interpolation
        if len(x_array) < 3:  # none + 2
            return {}  # cannot interpolate anything, return right away
        interp_func = scipy.interpolate.interp1d(
            x=np.asarray(x_array[1:]),
            y=np.asarray([np.mean(y_array[idx]) for idx in range(1, len(y_array))]),
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        missing_timestamps = interp_func(offset_distances[invalid_idxs])
        missing_timestamps = np.minimum(missing_timestamps, max_fb_timestamp)
        missing_timestamps = np.maximum(missing_timestamps, min_fb_timestamp)
        return {
            idx: timestamp for idx, timestamp in zip(invalid_idxs, missing_timestamps)
        }

    def _fill_missing_picks(self, gather: typing.Dict):
        """This function will fill the bad picks identified via their mask with proper estimates."""
        offset_distances, first_break_labels, first_break_timestamps, \
            to_fill_mask, valid_idxs, invalid_idxs = self._unpack_filter_vars(gather)
        if len(valid_idxs) == len(to_fill_mask):
            return  # all picks are valid, nothing to do here
        max_fb_timestamp, min_fb_timestamp = self.dataset.max_fb_timestamp, 1e-6
        if self.pick_fill_strategy == "full-linear-fit":
            missing_timestamps_map = self._compute_missing_picks_with_linear_fit(
                offset_distances=offset_distances,
                first_break_timestamps=first_break_timestamps,
                valid_idxs=valid_idxs,
                invalid_idxs=invalid_idxs,
                max_fb_timestamp=max_fb_timestamp,
                min_fb_timestamp=min_fb_timestamp,
            )
        else:  # pragma: no cover
            raise NotImplementedError
        # we've got our array of new values, run through it to validate/assign them
        filled_first_breaks_mask = np.full_like(to_fill_mask, False)
        for invalid_idx, missing_timestamp in missing_timestamps_map.items():
            if missing_timestamp > consts.BAD_FIRST_BREAK_PICK_INDEX:
                # note: the bad/outlier first breaks masks will not be updated!
                filled_first_breaks_mask[invalid_idx] = True
                first_break_timestamps[invalid_idx] = missing_timestamp
                missing_fb_label = (
                    missing_timestamp / max_fb_timestamp
                ) * self.dataset.samp_num
                missing_fb_label = int(
                    round(min(max(missing_fb_label, 0), self.dataset.samp_num - 1))
                )
                first_break_labels[invalid_idx] = missing_fb_label
        # store the newly created 'filled val mask' into the gather
        gather["filled_first_breaks_mask"] = filled_first_breaks_mask
        # the first break timestamps/labels above have been updated in-place, so we're done!
