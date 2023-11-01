"""Trace gather preprocessing module.

This module is meant to "preprocess" shot (line) gathers that are already parsed
and cleaned by other classes. Here, the preprocessing includes potential data augmentation
operations, trace amplitude normalization, and mask generation (for segmentation tasks).
"""
import functools
import logging
import typing

import numpy as np

from hardpicks.data.utils import get_value_or_default as vdefault
import hardpicks.data.fbp.gather_parser as gather_parser
import hardpicks.data.fbp.constants as consts
import hardpicks.data.fbp.gather_transforms as transforms
import hardpicks.data.transforms as generic_transforms
import hardpicks.models.constants as generic_consts

logger = logging.getLogger(__name__)


class ShotLineGatherPreprocessor(gather_parser.ShotLineGatherDatasetBase):
    """Wraps a shot (line) gather dataset parser object to preprocess its content.

    This object will behave exactly the same way as the gather dataset parser in the eyes of a
    data loader, but it will also filter/clean/postprocess the raw data that is read from the
    HDF5 files.

    It currently has five jobs: 1) apply amplitude normalization to trace samples based on various
    strategies; 2) generate segmentation masks based on first break labels to support segmentation
    models; 3) generate first break prior masks to help models determine which samples in a gather
    might actually hide the first breaks; 4) apply data augmentation operations (instantiated by
    name) on the parsed gathers; and 5) apply offset normalization across gathers based on various
    strategies. Each of these features can be individually deactivated based on the arguments given
    to the constructor.

    Attributes:
        dataset: reference to the dataset that provides the data to preprocess.
        normalize_samples: toggles whether sample amplitudes should be normalized or not.
        sample_norm_strategy: defines the normalization detection strategy to use.
        normalize_offsets: toggles whether offset distances should be normalized or not. For now,
            these can only be normalized with a constant factor that can also be user-provided.
        shot_to_rec_offset_norm_factor: defines the normalization factor to use when normalizing
            distances between shots and receivers; these distances are typically in [0-10]km range.
        rec_to_rec_offset_norm_factor: defines the normalization factor to use when normalizing
            distances between receivers; these distances are typically in [10-100]m range.
        generate_first_break_prior_masks: toggles whether first break prior masks should be
            generated or not. If toggled on, the following two arguments will also be used.
        first_break_prior_velocity_range: defines the (min,max) velocity values that will be used
            to generate the first break prior mask. These should be in km/s.
        first_break_prior_offset_range: defines the (min, max) offset or delay between the shot and
            the moment trace recording starts. This should be in milliseconds.
        generate_segm_masks: toggles whether segmentation masks should be generated or not.
        segm_class_count: defines the number of classes that the segmentation model will try to predict.
        segm_first_break_buffer: radius of the pixel buffer added around first breaks when
            generating segmentation masks.
        augmentations: a list of augmentation operations that should be applied to the data.
    """

    sample_normalization_strategies = [  # TODO add more strategies here if needed
        "tracewise-abs-max",  # normalize amplitudes for each trace independently with the abs-max
        # TODO "tracewise-mean-std",
        # TODO "trace-neighborhood-abs-max",
        # TODO "trace-neighborhood-mean-std",
        # TODO "gatherwise-...",
        # TODO ...
    ]

    # note: all the fields listed below will need to be padded in the collate function
    variable_length_fields = [  # each tuple is the field name + the default fill value
        ("segmentation_mask", generic_consts.DONTCARE_SEGM_MASK_LABEL),
        ("first_break_prior", 0.0),
    ]

    supported_augmentation_strategies = [  # TODO add more strategies here if needed
        "crop",  # see _augment_crop_samples below
        "resample_hardcoded",  # see _augment_resample_hardcoded below
        "resample_nearby",  # see _augment_resample_nearby below
        "drop_and_pad",  # see _augment_drop_and_pad_traces below
        "flip",  # will call the 'flip' transform
        "kill",  # will call the 'kill traces' transform
        "noise",  # will call the 'add_noise_patch' transform
    ]

    def __init__(
        self,
        dataset: gather_parser.ShotLineGatherDatasetBase,
        # these default to `None` so that config files can leave them empty and get real defaults below
        normalize_samples: typing.Optional[bool] = None,
        sample_norm_strategy: typing.Optional[typing.AnyStr] = None,
        normalize_offsets: typing.Optional[bool] = None,
        shot_to_rec_offset_norm_const: typing.Optional[int] = None,
        rec_to_rec_offset_norm_const: typing.Optional[int] = None,
        generate_first_break_prior_masks: typing.Optional[bool] = None,
        first_break_prior_velocity_range: typing.Optional[typing.Tuple[float, float]] = None,
        first_break_prior_offset_range: typing.Optional[typing.Tuple[float, float]] = None,
        generate_segm_masks: typing.Optional[bool] = None,
        segm_class_count: typing.Optional[int] = None,
        segm_first_break_buffer: typing.Optional[int] = None,
        augmentations: typing.Optional[typing.List] = None,
    ):
        """Validates and stores all the cleaning hyperparameters and a reference to the wrapped dataset."""
        self.dataset = dataset

        # default value = no, we will not normalize sample amplitudes
        self.normalize_samples = vdefault(normalize_samples, True)

        # default normalization strategy = trace-wise abs-max (as proposed by Gilles)
        assert self.normalize_samples or sample_norm_strategy is None, \
            "user specified a sample normalization strategy, but disabled sample normalization?"
        self.sample_norm_strategy = vdefault(sample_norm_strategy, "tracewise-abs-max")
        assert self.sample_norm_strategy in self.sample_normalization_strategies, \
            f"invalid normalization strategy: {sample_norm_strategy}"

        # default value = no, we will not normalize offset distances
        self.normalize_offsets = vdefault(normalize_offsets, True)

        # default shot-to-receiver normalization constant = 3000 (3km)
        assert self.normalize_offsets or shot_to_rec_offset_norm_const is None, \
            "user specified a distance normalization argument, but disabled distance normalization?"
        shot_to_rec_offset_norm_const = vdefault(shot_to_rec_offset_norm_const, 3000)
        assert shot_to_rec_offset_norm_const > 0, "invalid shot-to-rec offset const value"
        self.shot_to_rec_offset_norm_factor = 1 / shot_to_rec_offset_norm_const

        # default receiver-to-receiver normalization constant = 50 (50m)
        assert self.normalize_offsets or rec_to_rec_offset_norm_const is None, \
            "user specified a distance normalization argument, but disabled distance normalization?"
        rec_to_rec_offset_norm_const = vdefault(rec_to_rec_offset_norm_const, 50)
        assert rec_to_rec_offset_norm_const > 0, "invalid shot-to-rec offset const value"
        self.rec_to_rec_offset_norm_factor = 1 / rec_to_rec_offset_norm_const

        # default value = no, we will not generate the first break prior channel masks
        self.generate_first_break_prior_masks = vdefault(generate_first_break_prior_masks, False)
        assert self.generate_first_break_prior_masks or first_break_prior_velocity_range is None, \
            "user did not request first break priors but still specified an argument for it?"
        self.first_break_prior_velocity_range = \
            vdefault(first_break_prior_velocity_range, [4000, 7000])  # 4 km/s to 7 km/s
        assert self.generate_first_break_prior_masks or first_break_prior_offset_range is None, \
            "user did not request first break priors but still specified an argument for it?"
        self.first_break_prior_offset_range = \
            vdefault(first_break_prior_offset_range, [0, 40])  # 0 ms to 40 ms

        # default value = no, we will not generate segmentation masks
        self.generate_segm_masks = vdefault(generate_segm_masks, False)

        # by default, we assume that segmentation models work with a one-class setup
        self.segm_class_count = vdefault(segm_class_count, 1)
        assert self.segm_class_count in consts.SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP, \
            f"invalid class count for mask generation: {self.segm_class_count}"

        # by default, do not use a buffer to expand the first-break-class in the segm masks
        # note: it will only be used in the 1-class or 3-class setups
        assert self.generate_segm_masks or segm_first_break_buffer is None, \
            "user did not request segmentation masks but still specified an argument for them?"
        self.segm_first_break_buffer = vdefault(segm_first_break_buffer, 0)
        assert self.segm_first_break_buffer >= 0, "invalid first break buffer size (should be >= 0)"

        if augmentations is not None:
            augmentations = self._get_augmentation_ops(augmentations)
        self.augmentations = augmentations

    def _get_augmentation_ops(self, augmentation_config: typing.Iterable):
        """Returns a list of augmentation operations based on supported internal implementations."""
        assert isinstance(augmentation_config, list)
        assert all([isinstance(a, dict) for a in augmentation_config])
        aug_ops = []
        for aug_cfg in augmentation_config:
            assert aug_cfg["type"] in self.supported_augmentation_strategies
            if aug_cfg["type"] != "flip":
                if aug_cfg["type"] == "crop":
                    aug_cfg["type"] = self._augment_crop_samples
                elif aug_cfg["type"] == "resample_hardcoded":
                    aug_cfg["type"] = self._augment_resample_hardcoded
                elif aug_cfg["type"] == "resample_nearby":
                    aug_cfg["type"] = self._augment_resample_nearby
                elif aug_cfg["type"] == "drop_and_pad":
                    aug_cfg["type"] = self._augment_drop_and_pad_traces
                elif aug_cfg["type"] == "kill":
                    aug_cfg["type"] = transforms.kill_traces
                elif aug_cfg["type"] == "noise":
                    aug_cfg["type"] = transforms.add_noise_patch
                if "params" not in aug_cfg or not aug_cfg["params"]:
                    aug_cfg["params"] = {}  # just to make sure nothing blows up in the next line
                aug_ops.append(functools.partial(aug_cfg["type"], **aug_cfg["params"]))
            elif aug_cfg["type"] == "flip":
                # no params here, just need to wrap the operation w/ 0.5 prob
                aug_ops.append(generic_transforms.stochastic_op_wrapper(transforms.flip, 0.5))
            else:
                raise NotImplementedError
        return aug_ops

    def __len__(self) -> int:
        """Returns the total number of gathers in the wrapped dataset."""
        # this class should not need to change the number of samples in the underlying object...
        return len(self.dataset)

    def __getitem__(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing all pertinent information for a particular gather."""
        assert 0 <= gather_id < len(self.dataset), "gather query index is out-of-bounds"
        gather = self.dataset[gather_id]
        assert isinstance(gather, dict), "unexpected wrapped dataset gather type"
        if self.augmentations is not None:
            for augment in self.augmentations:
                augment(gather=gather)
        if self.generate_first_break_prior_masks:
            self._generate_first_break_prior_masks(gather)
        if self.normalize_samples:
            self._normalize_samples(gather)
        if self.normalize_offsets:
            self._normalize_offsets(gather)
        if self.generate_segm_masks:
            self._generate_segm_masks(gather)
        return gather

    def get_meta_gather(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing the meta data information.

        This method returns everything but the samples, and does not conduct the
        samples-dependent postprocessing that __getitem__ performs.
        """
        assert 0 <= gather_id < len(self.dataset), "gather query index is out-of-bounds"
        meta_gather = self.dataset.get_meta_gather(gather_id)
        return meta_gather

    def _normalize_samples(
        self,
        gather: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Normalizes the trace samples in the given gather with a particular strategy."""
        assert isinstance(gather, dict) and "samples" in gather
        samples = gather["samples"]
        assert len(samples.shape) == 2
        if self.sample_norm_strategy == "tracewise-abs-max":
            normed_samples = self.normalize_sample_with_tracewise_abs_max_strategy(samples)
        else:  # pragma: no cover
            raise NotImplementedError
        gather["samples"] = normed_samples

    @staticmethod
    def normalize_sample_with_tracewise_abs_max_strategy(samples):
        """Implement normalization of samples."""
        min_ampl_eps = 0.01  # better use an epsilon here for extra num stability
        tracewise_max_ampl = np.maximum(np.nanmax(np.abs(samples), axis=1), min_ampl_eps)
        assert not np.isnan(tracewise_max_ampl).any()
        normed_samples = samples / tracewise_max_ampl.reshape((-1, 1))
        return normed_samples

    def _normalize_offsets(
        self,
        gather: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Normalizes the offset distances in the given gather with constant factors."""
        assert isinstance(gather, dict) and "offset_distances" in gather, \
            "missing 'offset_distances' field in gathers (is 'provide_offset_dists' turned on?)"
        trace_count = gather["trace_count"]
        offsets = gather["offset_distances"]
        assert offsets.ndim == 2 and offsets.shape[0] == trace_count and offsets.shape[1] == 3
        offsets[:, 0] *= self.shot_to_rec_offset_norm_factor
        offsets[:, 1:] *= self.rec_to_rec_offset_norm_factor

    def _generate_segm_masks(
        self,
        gather: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Generates the segmentation masks for a given gather with a particular class count."""
        transforms.generate_segmentation_mask(
            gather,
            segm_class_count=self.segm_class_count,
            segm_first_break_buffer=self.segm_first_break_buffer,
        )

    def _generate_first_break_prior_masks(
        self,
        gather: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Generates the prior masks overlapping the region where first breaks should be found."""
        transforms.generate_prior_mask(
            gather,
            prior_velocity_range=self.first_break_prior_velocity_range,
            prior_offset_range=self.first_break_prior_offset_range,
        )

    @staticmethod
    def _augment_crop_samples(
        gather: typing.Dict[typing.AnyStr, typing.Any],
        low_sample_count: int,  # below this threshold, the gather will never be cropped
        high_sample_count: int,  # above this threshold, the gather will always be cropped
        max_crop_fraction: float = 0.333,  # we will NEVER crop more than this % of the samples
    ):
        """Crops the sample axis in a gather to make it 'shorter'. Do not use in eval mode!"""
        assert 0 < low_sample_count < high_sample_count
        assert 0 < max_crop_fraction < 1
        sample_count = gather["sample_count"]
        if sample_count <= low_sample_count:
            return  # do nothing, the gather is already too small to be cropped
        hard_max_crop_count = int(round(sample_count * max_crop_fraction))
        assert hard_max_crop_count < sample_count
        max_crop_count = min(sample_count - low_sample_count, hard_max_crop_count)
        min_crop_count = min(max(sample_count - high_sample_count, 0), hard_max_crop_count)
        if min_crop_count >= max_crop_count:
            crop_count = min_crop_count
        else:
            crop_count = np.random.randint(min_crop_count, max_crop_count + 1)
        assert sample_count - crop_count >= low_sample_count
        assert crop_count <= hard_max_crop_count
        transforms.crop_samples(gather, sample_count - crop_count)

    @staticmethod
    def _augment_resample_hardcoded(
        gather: typing.Dict[typing.AnyStr, typing.Any],
        low_sample_count: int,  # the gather will never be resampled to a length below this threshold
        high_sample_count: int,  # the gather will never be resampled to a length above this threshold
        downsample_only: bool,  # might be less crazy that way, i.e. we don't try to hallucinate details
        allowed_sample_rates: typing.Optional[typing.Sequence] = None,  # none = default ([0.5, 1.0, 2.0, 4.0])
    ):
        """Resamples the gather at a rate in [0.5ms, 1ms, 2ms, 4ms] while respecting thresholds."""
        # note1: we will only ever resample one-sample-rate-away (e.g. 0.5ms to 1ms, but never 0.5ms to 2ms)
        # note2: if the low/high sample counts disallow a chosen sample rate, the resampling will be skipped
        assert 0 < low_sample_count < high_sample_count
        if not allowed_sample_rates:
            allowed_sample_rates = [0.5, 1.0, 2.0, 4.0]
        assert len(allowed_sample_rates) > 0
        curr_sample_rate = gather["sample_rate_ms"]
        potential_new_sample_rates = []
        if curr_sample_rate in allowed_sample_rates:
            curr_sample_rate_idx = allowed_sample_rates.index(curr_sample_rate)
            if curr_sample_rate_idx > 0 and not downsample_only:
                potential_new_sample_rates.append(allowed_sample_rates[curr_sample_rate_idx - 1])
            if curr_sample_rate_idx < len(allowed_sample_rates) - 1:
                potential_new_sample_rates.append(allowed_sample_rates[curr_sample_rate_idx + 1])
        else:
            allowed_sample_rates = np.asarray(allowed_sample_rates)
            if not downsample_only:
                up_sample_rates = allowed_sample_rates[allowed_sample_rates < curr_sample_rate]
                if len(up_sample_rates):
                    potential_new_sample_rates.append(up_sample_rates.max())
            down_sample_rates = allowed_sample_rates[allowed_sample_rates > curr_sample_rate]
            if len(down_sample_rates):
                potential_new_sample_rates.append(down_sample_rates.min())
        new_sample_rate = np.random.choice(potential_new_sample_rates)
        curr_sample_count = gather["sample_count"]
        new_sample_count = int(round((curr_sample_rate / new_sample_rate) * curr_sample_count))
        if new_sample_count < low_sample_count or new_sample_rate > high_sample_count:
            return  # skip the resampling if it would bust the low/high sample count thresholds
        transforms.resample(gather, new_sample_count)

    @staticmethod
    def _augment_resample_nearby(
        gather: typing.Dict[typing.AnyStr, typing.Any],
        prob: float,  # the probability of applying this augmentation
        low_sample_count: int,  # the gather will never be resampled to a length below this threshold
        high_sample_count: int,  # the gather will never be resampled to a length above this threshold
        downsample_only: bool,  # might be less crazy that way, i.e. we don't try to hallucinate details
        sample_rate_win_size: float,  # the size of the uniform sampling window around the current rate
        min_sample_rate: float = 0.5,
        max_sample_rate: float = 4.0,
    ):
        """Resamples the gather at a rate near the current sample rate while respecting thresholds."""
        # note: the low/high sample counts here will act as min/max sample counts (can skew statistics!)
        assert 0 <= prob <= 1
        if np.isclose(prob, 0.0) or np.random.rand() > prob:
            return
        assert 0 < low_sample_count < high_sample_count
        assert sample_rate_win_size > 0
        curr_sample_rate = gather["sample_rate_ms"]
        if downsample_only:
            sample_rate_var = np.random.uniform(0, sample_rate_win_size)
        else:
            sample_rate_var = np.random.uniform(-sample_rate_win_size, sample_rate_win_size)
        new_sample_rate = max(min(curr_sample_rate + sample_rate_var, max_sample_rate), min_sample_rate)
        curr_sample_count = gather["sample_count"]
        new_sample_count = int(round((curr_sample_rate / new_sample_rate) * curr_sample_count))
        new_sample_count = max(min(new_sample_count, high_sample_count), low_sample_count)
        transforms.resample(gather, new_sample_count)

    @staticmethod
    def _augment_drop_and_pad_traces(
        gather: typing.Dict[typing.AnyStr, typing.Any],
        target_trace_counts: typing.Sequence[int],
        full_snap: bool,  # if toggled on, will always full pad/drop to target count
        max_drop_ratio: float = 0.25,  # max fraction of traces that can be dropped to reach target
    ):
        """Drops traces with bad picks at gather edges and/or pads gathers towards a target size."""
        # note1: this is done to increase the likelihood of fitting multiple gathers in the same range
        # note2: the padding/trace drops will aim to bring the trace towards (or on) the closest target
        curr_trace_count = gather["trace_count"]
        assert len(target_trace_counts) > 0
        # sort the target trace counts array and figure out which one to aim for
        target_trace_counts = np.sort(np.asarray(target_trace_counts))
        target_trace_count_idx = np.argmin(np.abs(target_trace_counts - curr_trace_count))
        target_trace_count = target_trace_counts[target_trace_count_idx]
        # determine exactly how far (and in what direction) we are from that target trace count
        trace_count_var = target_trace_count - curr_trace_count
        max_drop_count = int(round(max_drop_ratio * curr_trace_count))
        # check whether we'd drop too many traces to actually reach that target...
        if trace_count_var < 0 and abs(trace_count_var) > max_drop_count:
            # if so, we need to switch over to the next (bigger) target (and pad instead of drop)
            assert target_trace_count_idx < len(target_trace_counts) - 1, \
                f"gather too big for the current max limit in 'drop-and-pad'" \
                f"(curr={curr_trace_count}, limit={target_trace_counts[-1]})"
            target_trace_count = target_trace_counts[target_trace_count_idx + 1]
            trace_count_var = target_trace_count - curr_trace_count
        if trace_count_var < 0:
            assert abs(trace_count_var) <= max_drop_count
            # if we get here, we actually need to drop some traces (finally!)
            if not full_snap:
                # if we don't need to absolutely reach the goal, just get a distance towards it
                trace_count_var = np.random.randint(abs(trace_count_var) + 1)
            transforms.drop_traces(gather, abs(trace_count_var), True, True)
        elif trace_count_var > 0:
            # otherwise, if we get here, we need to pad with dummy traces (distrib on both side!)
            if not full_snap:
                trace_count_var = np.random.randint(trace_count_var + 1)
            prepad_size = np.random.randint(trace_count_var)
            postpad_size = trace_count_var - prepad_size
            transforms.pad_traces(gather, prepad_size, postpad_size)
