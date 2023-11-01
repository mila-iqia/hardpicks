"""Trace gather parsing module."""

import logging
import os
import typing

import h5py
import numpy as np
import torch.utils.data

import hardpicks.data.fbp.trace_parser as trace_parser
import hardpicks.data.fbp.constants as consts

logger = logging.getLogger(__name__)


class ShotLineGatherDatasetBase(torch.utils.data.Dataset):
    """Base class for all gather parsing modules.

    Other classes that wrap the actual dataset parsing module (`ShotLineGatherDataset`) will need
    to implement all functions in this interface in order to truly be "pass-through" in the eyes
    of data loaders and other wrapping modules.
    """

    def __len__(self) -> int:
        """Returns the total number of gathers in the dataset."""
        raise NotImplementedError

    def __getitem__(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing all pertinent information for a particular gather."""
        raise NotImplementedError

    def get_meta_gather(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing all meta information for a particular gather."""
        # note: compared to the __getitem__ call, this call should be lightweight and only based
        #       on readily available data (no loading from disk --- it needs to be FAST!)
        raise NotImplementedError


class ShotLineGatherDataset(ShotLineGatherDatasetBase, trace_parser.RawTraceDataset):
    """Container for raw FBP shot gather data read directly from an HDF5 file.

    The `size` of this container will be equal to the size of the intersection between shots and
    receiver lines. Its `__getitem__` function will return each 'gather' one-by-one as a dictionary
    of all useful fields. Since this class does not allow multiple lines to be combined in a gather,
    the features that the model will have to ingest with this dataset is a 2D tensor.
    """

    # note: all the fields listed below will need to be padded in the collate function
    variable_length_fields = [  # each tuple is the field name + the default fill value
        ("rec_ids", consts.BAD_OR_PADDED_ELEMENT_ID),
        ("gather_trace_ids", consts.BAD_OR_PADDED_ELEMENT_ID),
        ("first_break_labels", consts.BAD_FIRST_BREAK_PICK_INDEX),
        ("first_break_timestamps", consts.BAD_FIRST_BREAK_PICK_INDEX),
        ("bad_first_breaks_mask", True),
        ("dead_rec_mask", True),
        ("rec_coords", 0),
        ("offset_distances", 0),
        ("samples", 0),
    ]

    def __init__(
        self,
        hdf5_path: typing.AnyStr,
        site_name: typing.AnyStr,  # the 'name' of the site (origin) that will be given to all items
        receiver_id_digit_count: int,  # the digit count allocated for IDs in the rec peg number
        first_break_field_name: typing.AnyStr,  # the variable name in the HDF5 file where the fbp are stored.
        convert_to_fp16: bool,  # will convert fp32 to fp16 where possible to save memory
        convert_to_int16: bool,  # will convert uint32/int32 to int16 where possible to save memory
        preload_trace_data: bool,  # toggles whether all trace data should be pre-loaded or not
        cache_trace_metadata: bool,  # toggles whether metadata should be computed once and cached
        provide_offset_dists: bool,  # toggles whether to compute the offset distances array
    ):
        """Nothing special to do in here, just forwards all args to the base class constructor."""
        super().__init__(
            hdf5_path=hdf5_path,
            receiver_id_digit_count=receiver_id_digit_count,
            first_break_field_name=first_break_field_name,
            convert_to_fp16=convert_to_fp16,
            convert_to_int16=convert_to_int16,
            preload_trace_data=preload_trace_data,
        )
        self.site_name = site_name
        self.metadata_cache = {}
        self.cache_trace_metadata = cache_trace_metadata
        self.provide_offset_dists = provide_offset_dists

    def _get_dead_rec_mask(self, trace_array: np.ndarray) -> np.ndarray:
        """Returns a binary mask that indicates which receivers are dead in the provided trace array."""
        assert trace_array.ndim == 2 and trace_array.shape[1] == self.samp_num, "unexpected array shape"
        return np.isclose(trace_array, 0, atol=consts.DEAD_TRACE_AMPLITUDE_TOLERANCE).all(axis=1)

    @staticmethod
    def _get_bad_first_breaks_mask(fb_labels: np.ndarray, fb_tstamps: np.ndarray) -> np.ndarray:
        """Returns a binary mask that indicates which receivers have bad first break labels."""
        assert np.array_equal(np.where(fb_labels <= consts.BAD_FIRST_BREAK_PICK_INDEX),
                              np.where(fb_tstamps <= consts.BAD_FIRST_BREAK_PICK_INDEX)), \
            "unexpected mismatch between fb labels and timestamps?"
        return np.where(fb_labels == -1, True, False)

    def _get_offset_distances_array(self, rec_ids: typing.Sequence[int], shot_id: int) -> np.ndarray:
        """Returns the array of offset distances between receivers-shots and receiver-receiver (x2)."""
        rec_coords = np.asarray([coords for rec_id in rec_ids
                                 for coords in self.rec_to_coords_map[rec_id]]).reshape(-1, 3)
        shot_coords = self.shot_to_coords_map[shot_id]
        computed_shot_offsets = np.asarray([np.linalg.norm(r - shot_coords) for r in rec_coords])
        if len(computed_shot_offsets) > 1:
            rec_diff_offsets = np.asarray([np.linalg.norm(r1 - r2) for r1, r2 in zip(rec_coords, rec_coords[1:])])
            rec_next_offsets = np.concatenate((rec_diff_offsets, [0]))
            rec_prev_offsets = np.concatenate(([0], rec_diff_offsets))
        else:
            rec_next_offsets = np.full_like(computed_shot_offsets, 0)
            rec_prev_offsets = np.full_like(computed_shot_offsets, 0)
        return np.stack((computed_shot_offsets, rec_next_offsets, rec_prev_offsets), axis=1)

    def _is_gather_valid(self, gather_id):
        """Returns whether a gather is 'valid' (i.e. it has at least one non-bad pick)."""
        assert 0 <= gather_id < len(self.gather_to_trace_map), "gather query index is out-of-bounds"
        gather_trace_ids = self.gather_to_trace_map[gather_id]
        bad_first_breaks_mask = self._get_bad_first_breaks_mask(
            self.first_break_labels[gather_trace_ids], self.first_break_timestamps[gather_trace_ids])
        return not bad_first_breaks_mask.all()

    def __len__(self) -> int:
        """Returns the total number of gathers in the dataset."""
        return len(self.gather_to_trace_map)

    def get_meta_gather(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing all meta information for a particular gather."""
        if self.cache_trace_metadata and gather_id in self.metadata_cache:
            return self.metadata_cache[gather_id]
        assert 0 <= gather_id < len(self.gather_to_trace_map), "gather query index is out-of-bounds"
        gather_trace_ids = self.gather_to_trace_map[gather_id]
        # just get the id of the first trace to fetch values that are constant across the shot gather
        first_trace_id = gather_trace_ids[0]
        rec_ids = self.trace_to_rec_map[gather_trace_ids]
        shot_id = self.trace_to_shot_map[first_trace_id]
        offset_distances = None
        if self.provide_offset_dists:
            offset_distances = self._get_offset_distances_array(rec_ids, shot_id)
        bad_first_breaks_mask = self._get_bad_first_breaks_mask(
            self.first_break_labels[gather_trace_ids], self.first_break_timestamps[gather_trace_ids])
        gather_metadata = dict(
            origin=self.site_name,
            shot_id=shot_id,
            rec_line_id=self.trace_to_line_map[first_trace_id],
            rec_ids=rec_ids,
            gather_id=gather_id,
            gather_trace_ids=gather_trace_ids,
            first_break_labels=self.first_break_labels[gather_trace_ids],
            first_break_timestamps=self.first_break_timestamps[gather_trace_ids],
            bad_first_breaks_mask=bad_first_breaks_mask,
            rec_coords=np.stack([self.rec_to_coords_map[id] for id in rec_ids]),
            shot_coords=self.shot_to_coords_map[shot_id],
            offset_distances=offset_distances,
            trace_count=len(rec_ids),  # kept here for easier debugging after collating
            sample_count=int(self.samp_num),  # kept here for easier debugging after collating
            sample_rate_ms=self.samp_rate / 1000,  # may be changed during data augmentation
        )
        if self.cache_trace_metadata:
            self.metadata_cache[gather_id] = gather_metadata
        return gather_metadata

    def __getitem__(self, gather_id: int) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Returns a dictionary containing all pertinent information for a particular gather."""
        meta_gather = self.get_meta_gather(gather_id)
        trace_ids = meta_gather["gather_trace_ids"]
        if self.preload_trace_data:
            samples = self.data_array[trace_ids]
        else:
            if self._worker_h5fd is None:
                self._worker_h5fd = h5py.File(self.hdf5_path, mode="r")
            samples = self._worker_h5fd["TRACE_DATA"]["DEFAULT"]["data_array"][trace_ids]
        return {
            **meta_gather,
            "dead_rec_mask": self._get_dead_rec_mask(samples),
            "samples": samples,
        }


def create_shot_line_gather_dataset(
    hdf5_path: typing.AnyStr,
    site_name: typing.AnyStr,
    receiver_id_digit_count: int,  # could change from one site to the other
    first_break_field_name: typing.AnyStr,  # the variable name in the HDF5 file where the fbp are stored.
    convert_to_fp16: bool = True,  # we don't need full fp32 precision by default
    convert_to_int16: bool = True,  # we don't need full int32/uint32 precision by default
    preload_trace_data: bool = False,  # might be useful on machines with lots of RAM
    cache_trace_metadata: bool = False,  # toggles whether metadata should be computed once and cached
    provide_offset_dists: bool = False,  # will be useful for learning geometry-aware stuff
) -> ShotLineGatherDataset:
    """Reads and returns the shot gather data contained in an HDF5 archive for first break picking.

    Also does a whole bunch of input validation and cleanup in the process.

    Note: this parsing function assumes that all fields are the same across sites, and
    identical to the ones in the Halfmile_3D archive.
    """
    assert os.path.isfile(hdf5_path), f"cannot open hdf5 archive at path: {hdf5_path}"
    logger.info(f"parsing HDF5 data from: {hdf5_path}")
    dataset = ShotLineGatherDataset(
        hdf5_path,
        site_name=site_name,
        receiver_id_digit_count=receiver_id_digit_count,
        first_break_field_name=first_break_field_name,
        convert_to_fp16=convert_to_fp16,
        convert_to_int16=convert_to_int16,
        preload_trace_data=preload_trace_data,
        cache_trace_metadata=cache_trace_metadata,
        provide_offset_dists=provide_offset_dists,
    )
    return dataset
