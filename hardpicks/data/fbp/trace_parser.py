"""HDF5 trace parsing module.

After a discussion with our NRCAN colleagues on 2021/02/18, the following HDF5 fields have
been identified as 'potentially useful' for our first break picking project:
  OFFSET, REC_HT, REC_PEG, REC_X, REC_Y, SAMP_NUM, SAMP_RATE, SHOTID, SHOT_PEG, SOURCE_HT,
  SOURCE_X, SOURCE_Y, SPARE1, data_array

More info: https://docs.google.com/document/d/1QVEsWh7x3xpZW7I2V6ix6m7ifVA_Ivo8bJ_dw4_EL98

We will assume that HDF5 files for all sites contain the same fields.

Note: some of the content of this file overlaps with the `FirstBreakPickingSeismicData` class
in the `analysis` subpackage. This is known, and we'll keep an eye on both to make sure the
decision to keep them separated remains ideal.

Update 2021/06/04:  It will be useful to use different SPARE* variables to store different kinds
of picks. Correspondingly, we will no longer assume that SPARE1 is the only possibility.
"""

import logging
import os
import typing

import h5py
import numpy as np
from torch.utils.data import Dataset

import hardpicks.data.fbp.constants as consts

logger = logging.getLogger(__name__)

BASE_EXPECTED_HDF5_FIELDS = [
    # note: not all of these will be used right away, we will revisit the loader later
    "REC_PEG",
    "REC_X",
    "REC_Y",
    "REC_HT",
    "SAMP_NUM",
    "SAMP_RATE",
    "COORD_SCALE",
    "HT_SCALE",
    "SHOTID",
    "SHOT_PEG",
    "SOURCE_X",
    "SOURCE_Y",
    "SOURCE_HT",
    "data_array",
]


class RawTraceDataset(Dataset):
    """Container for raw FBP trace data read directly from an HDF5 file.

    The `size` of this container will be equal to the total number of traces in the dataset
    (pre-cleaning/filtering). Its `__getitem__` function will return each trace one-by-one
    as a dictionary of all useful fields.

    Attributes:
        receiver_id_digit_count: defines how many digits will be imputed to the receiver ids in
            the encoded receiver peg number.
        first_break_field_name: The variable name in the HDF5 file where the fbp are stored.
        convert_to_fp16: defines whether fp32 values will converted to fp16 values to save memory.
        convert_to_int16: defines whether int32 values will converted to int16 values to save memory.
        total_trace_count: the total number of traces (equal to the number of deployed geophones).
        trace_to_shot_map: a map that provides the unique shot id associated with each trace in
            the dataset.
        shot_to_trace_map: a map that provides the array of trace ids associated with each shot
            in the dataset.
        trace_to_line_map: a map that provides the unique receiver line id associated with each
            trace in the dataset.
        line_to_trace_map: a map that provides the array of trace ids associated with each receiver
            line in the dataset.
        trace_to_rec_map: a map that provides the unique receiver id associated with each trace in
            the dataset.
        rec_to_trace_map: a map that provides the array of trace ids associated with each receiver
            in the dataset.
        trace_to_gather_map: a map that provides the unique gather id associated with each trace in
            the dataset.
        gather_to_trace_map: a map that provides the array of trace ids associated with each gather
            in the dataset.
        samp_num: the number of samples per trace.
        samp_rate: the sampling rate of receivers (in microseconds).
        first_break_labels: the sample indices that were picked as the 'first break' by an
            annotator or automated tool. There is one value per trace.
        first_break_timestamps: the timestamps (in milliseconds) that were picked as the 'first
            break' by an annotator or automated tool. There is one value per trace.
        coord_scale: scaling factor to be applied to the parsed receiver/shot X and Y coordinates to
            get their real (UTM?) coordinates.
        ht_scale: scaling factor to be applied to the parsed receiver/shot Z (height) coordinate to
            get their real (datum-relative? sea-level-offset?) coordinates.
        shot_to_coords_map: a map that provides the 3d coordinates (X,Y,Z) associated to each shot
            in the dataset.
        rec_to_coords_map: a map that provides the 3d coordinates (X,Y,Z) associated to each
            receiver (geophone) in the dataset.
        data_array: the raw trace data array. Only available as an attribute if the dataset is
            preloaded.
        preload_trace_data: defines whether to pre-load the trace array or to defer it to the
            sample loader. In the latter case, a file handle will be kept for future calls.
    """

    def __init__(
        self,
        hdf5_path: typing.AnyStr,
        receiver_id_digit_count: int,  # the digit count allocated for IDs in the rec peg number
        first_break_field_name: typing.AnyStr,  # the variable name in the HDF5 file where the fbp are stored.
        convert_to_fp16: bool,  # will convert fp32 to fp16 where possible to save memory
        convert_to_int16: bool,  # will convert uint32/int32 to int16 where possible to save memory
        preload_trace_data: bool,  # toggles whether all trace data should be pre-loaded or not
    ):
        """Building the object will actually open and parse the hdf5 archive."""
        self._validate_first_break_field_name(first_break_field_name)

        expected_hdf5_fields = BASE_EXPECTED_HDF5_FIELDS + [first_break_field_name]

        logger.debug(f"parsing hdf5 file at: {hdf5_path}")
        with h5py.File(hdf5_path, mode="r") as h5root:
            self._check_h5data_struct(h5root, expected_hdf5_fields)

            logger.debug("parsing dataset hyperparameters...")
            h5data = h5root["TRACE_DATA"]["DEFAULT"]
            self.receiver_id_digit_count = receiver_id_digit_count
            self.first_break_field_name = first_break_field_name
            self.convert_to_fp16 = convert_to_fp16
            self.convert_to_int16 = convert_to_int16
            self.preload_trace_data = preload_trace_data
            self.hdf5_path = hdf5_path

            logger.debug("parsing dataset global constants...")
            self.samp_num = self._get_const_parameter(h5data, "SAMP_NUM")
            self.samp_rate = self._get_const_parameter(h5data, "SAMP_RATE")  # in usec (typically?)
            self.max_fb_timestamp = self.samp_num * self.samp_rate / 1000  # in msec (based on above?)
            self.coord_scale = self._get_const_parameter(h5data, "COORD_SCALE")
            self.ht_scale = self._get_const_parameter(h5data, "HT_SCALE")
            self.total_trace_count = len(h5data["data_array"])
            logger.debug(f"found {self.total_trace_count} traces")

            logger.debug("parsing dataset shot/receiver/gather maps...")
            self.trace_to_shot_map, self.shot_to_trace_map = self._get_id_maps(
                h5data, "SHOT_PEG", decode_peg_id=True, id_digit_count=0
            )  # no need for digit count
            logger.debug(f"found {len(self.shot_to_trace_map)} shots")
            self.trace_to_rec_map, self.rec_to_trace_map = self._get_id_maps(
                h5data,
                "REC_PEG",
                decode_peg_id=False,
                id_digit_count=receiver_id_digit_count,
            )
            logger.debug(f"found {len(self.rec_to_trace_map)} receivers")
            self.trace_to_line_map, self.line_to_trace_map = self._get_id_maps(
                h5data,
                "REC_PEG",
                decode_peg_id=True,
                id_digit_count=receiver_id_digit_count,
            )
            logger.debug(f"found {len(self.line_to_trace_map)} lines")
            self.trace_to_gather_map, self.gather_to_trace_map = self._get_gather_maps()
            logger.debug(
                f"found {len(self.gather_to_trace_map)} gather (i.e. valid lines x shots)"
            )

            logger.debug("parsing dataset first break annotations...")
            self.first_break_labels, self.first_break_timestamps = self._get_first_breaks(
                h5data, self.first_break_field_name
            )

            logger.debug("parsing dataset receivers/shots geometry...")
            self.rec_to_coords_map = self._get_coords(h5data, "REC")
            self.shot_to_coords_map = self._get_coords(h5data, "SHOT")

            if self.preload_trace_data:
                logger.debug("parsing dataset trace data...")
                self.data_array = self._get_trace_data(h5data)
            else:
                self._worker_h5fd = None

        logger.debug("parsing complete!")

    @staticmethod
    def _validate_first_break_field_name(first_break_field_name: str):
        valid_names = ['SPARE1', 'SPARE2', 'SPARE3', 'SPARE4']
        assert first_break_field_name in valid_names, 'the first break field name is wrong.'

    @staticmethod
    def _check_h5data_struct(h5root: h5py.Group, expected_hdf5_fields: typing.List[str]) -> None:
        """Parses the top-level hdf5 objects and validates the dataset structure."""
        assert "TRACE_DATA" in h5root, "unexpected root level hdf5 structure"
        assert "DEFAULT" in h5root["TRACE_DATA"], "unexpected root level hdf5 structure"
        h5data = h5root["TRACE_DATA"]["DEFAULT"]
        expected_trace_count = None
        for expected_field in expected_hdf5_fields:
            assert (
                expected_field in h5data
            ), f"missing field {expected_field} in HDF5 file"
            if expected_trace_count is None:
                expected_trace_count = len(h5data[expected_field])
            else:
                assert expected_trace_count == len(h5data[expected_field]), (
                    f"unexpected dataset size for field {expected_field}\n"
                    f"\t({len(h5data[expected_field])} instead of {expected_trace_count})"
                )

    def _get_id_maps(
        self,
        h5data: h5py.Group,
        peg_name: typing.AnyStr,
        decode_peg_id: bool,
        id_digit_count: int,
    ) -> typing.Tuple[typing.Sequence[int], typing.Dict[int, typing.Sequence[int]]]:
        """Returns both the trace-to-id and id-to-trace mappings for receivers or shots."""
        assert peg_name in ["SHOT_PEG", "REC_PEG"], "unexpected peg type"
        # here, we drop the station id and keep only the shot id, saving a bit of memory...
        # Convert to np.int32 from np.uint32 because pytorch cannot handle unsigned integers:
        # since we'll feed this into pytorch eventually, it's better to snip this problem in the bud.
        pegs = np.array(h5data[peg_name]).flatten().astype(np.int32)
        if decode_peg_id:
            if peg_name == "SHOT_PEG":
                if np.unique(h5data["SHOT_PEG"]).size == 1:
                    # pegs are sometimes broken (e.g. in the sudbury site), swap to SHOTID in that case
                    pegs = np.array(h5data["SHOTID"]).flatten().astype(np.int32)
                else:
                    assert (
                        np.unique(h5data["SHOT_PEG"]).size
                        == np.unique(h5data["SHOTID"]).size
                    )
                ids = (
                    pegs
                )  # for shots, we can keep the full peg as the identifier (same result)
            else:  # peg_name == "REC_PEG"
                # for receiver pegs, the ID we want to keep is only the first few digits
                ids = pegs // (10 ** id_digit_count)
                if self.convert_to_int16:
                    assert (
                        ids.max() < np.iinfo(np.int16).max
                    ), f"oob {peg_name} for 16-bit conversion"
                    ids = ids.astype(np.int16)
        else:
            ids = pegs
        reversed_map = {
            sid: np.where(ids == sid)[0].astype(np.int32) for sid in np.unique(ids)
        }
        assert sum([len(idxs) for idxs in reversed_map.values()]) == len(
            pegs
        ), "bad reverse map impl"
        return ids, reversed_map

    def _get_gather_maps(
        self,
    ) -> typing.Tuple[typing.Sequence, typing.Dict[int, typing.Sequence[int]]]:
        """Returns both the trace-to-gather and gather-to-trace mappings for the dataset."""
        gather_to_trace_map = {}
        trace_to_gather_map = [None] * self.total_trace_count
        # first, double-loop on shot+traces to find intersections
        for shot_id, shot_traces in self.shot_to_trace_map.items():
            for line_id, line_traces in self.line_to_trace_map.items():
                gather_traces = np.intersect1d(shot_traces, line_traces)
                if not len(gather_traces):
                    continue
                next_gather_id = len(gather_to_trace_map)
                gather_to_trace_map[next_gather_id] = gather_traces
        # now, reloop over all gathers to assign ids in the direct map
        for gather_id, gather_traces in gather_to_trace_map.items():
            for trace_id in gather_traces:
                assert (
                    0 <= trace_id < self.total_trace_count
                ), "bad trace id found in gather?"
                assert (
                    trace_to_gather_map[trace_id] is None
                ), "trace had multiple intersections?"
                trace_to_gather_map[trace_id] = gather_id
        assert not any(
            [t is None for t in trace_to_gather_map]
        ), "there are useless traces?"
        return trace_to_gather_map, gather_to_trace_map

    @staticmethod
    def _get_const_parameter(
        h5data: h5py.Group, field_name: typing.AnyStr
    ) -> np.number:
        """Returns a constant-across-all-traces parameter from the HDF5 group."""
        data = np.unique(h5data[field_name])
        assert len(data) == 1, f"invalid field {field_name} (content not unique)"
        return data[0]

    def _get_first_breaks(
        self,
        h5data: h5py.Group,
        field_name: typing.AnyStr = "SPARE1",  # it's a strange field name, but it seems standard
    ) -> typing.Tuple[typing.Sequence[int], typing.Sequence[float]]:
        """Returns the maps of label indices and timestamps for first break picks."""
        assert (
            self.samp_num > 0 and self.samp_rate > 0
        ), "invalid sample rate/count values"
        assert (
            self.samp_num == h5data["data_array"].shape[-1]
        ), "bad trace data sample count"
        assert (
            self.samp_rate >= 1000
        ), "current impl assumes rate is provided in usec, still true?"
        # we'll assume that fddata contains traces in msec, so we have to use the rate to convert
        fbdata_msec = np.array(h5data[field_name]).flatten()
        assert (
            len(fbdata_msec) == self.total_trace_count
        ), "unexpected first break data array shape"
        samp_rate_msec = self.samp_rate / 1000.0
        bad_fb_picks = np.where(fbdata_msec <= consts.BAD_FIRST_BREAK_PICK_INDEX)
        fbdata_idxs = self._get_first_break_indices(fbdata_msec, samp_rate_msec)
        fbdata_idxs[bad_fb_picks] = -1
        assert (fbdata_idxs < self.samp_num).all(), "are picks really provided in msec?"
        if self.convert_to_int16:
            fbdata_idxs = fbdata_idxs.astype(np.int16)
        if self.convert_to_fp16:
            fbdata_msec = fbdata_msec.astype(np.float16)
        return fbdata_idxs, fbdata_msec

    @staticmethod
    def _get_first_break_indices(
        fbp_times_in_milliseconds: np.ndarray,
        sample_rate_in_milliseconds: float,
    ):
        ratio = fbp_times_in_milliseconds / sample_rate_in_milliseconds
        small_ratio_mask = np.bitwise_and(ratio > 0., ratio <= 1.)
        fbp_indices = np.floor(ratio)
        # make sure that small first break picks are not assigned to zero, which
        # we flag as abnormal.
        fbp_indices[small_ratio_mask] = 1
        return fbp_indices.astype(np.int32)

    def _get_coords(
        self, h5data: h5py.Group, target_set: typing.AnyStr
    ) -> typing.Dict[int, typing.Sequence[float]]:
        """Returns the map of coordinates for either receivers or shots (based on `target_set`)."""
        assert target_set in ["REC", "SHOT"], "unexpected coords target set"
        if target_set == "REC":
            target_fields = ["REC_X", "REC_Y", "REC_HT"]
            target_map = self.rec_to_trace_map
        else:
            target_fields = ["SOURCE_X", "SOURCE_Y", "SOURCE_HT"]
            target_map = self.shot_to_trace_map
        coords = np.stack(
            [np.array(h5data[field]).flatten() for field in target_fields], axis=1
        )
        # the demo notebooks take the abs val of all scaling factors, so we'll do the same
        xy_scale, z_scale = (
            np.abs(float(self.coord_scale)),
            np.abs(float(self.ht_scale)),
        )
        coords = (coords / np.asarray((xy_scale, xy_scale, z_scale))).astype(np.float32)
        out_coords_map = {}
        for target_id, trace_ids in target_map.items():
            unique_coords = np.unique(coords[trace_ids], axis=0)
            if len(unique_coords) > 1:
                # yes, it can sometimes happen, we just need to make sure it's only due to fp error
                assert [
                    np.allclose(unique_coords[0], unique_coords[i])
                    for i in range(1, len(unique_coords))
                ], "unexpected coordinate variation inside group"
            out_coords_map[target_id] = unique_coords[0]
        return out_coords_map

    def _get_trace_data(self, h5data: h5py.Group) -> typing.Sequence[float]:
        """Returns the 2D array of trace samples (converting on-the-fly if necessary)."""
        # note: this might take a while without caching for large datasets!
        if self.convert_to_fp16:
            prealloc_array = np.empty(
                (self.total_trace_count, self.samp_num), np.float16
            )
        else:
            prealloc_array = np.empty(
                (self.total_trace_count, self.samp_num), np.float32
            )
        h5data["data_array"].read_direct(prealloc_array)
        return prealloc_array

    def __len__(self):
        """Returns the total number of traces in the dataset."""
        return self.total_trace_count

    def __getitem__(self, trace_id):
        """Returns a dictionary containing all pertinent information for a particular trace."""
        assert (
            0 <= trace_id < self.total_trace_count
        ), "trace query index is out-of-bounds"
        # for efficient gather-level parsing, a derived class should bypass this call entirely...
        rec_id = self.trace_to_rec_map[trace_id]
        shot_id = self.trace_to_shot_map[trace_id]
        if self.preload_trace_data:
            samples = self.data_array[trace_id]
        else:
            if self._worker_h5fd is None:
                self._worker_h5fd = h5py.File(self.hdf5_path, mode="r")
            samples = self._worker_h5fd["TRACE_DATA"]["DEFAULT"]["data_array"][trace_id]
        assert len(samples) == self.samp_num
        return dict(
            trace_id=trace_id,
            shot_id=shot_id,
            rec_line_id=self.trace_to_line_map[trace_id],
            rec_id=rec_id,
            gather_id=self.trace_to_gather_map[trace_id],
            first_break_label=self.first_break_labels[trace_id],
            first_break_timestamp=self.first_break_timestamps[trace_id],
            rec_coords=self.rec_to_coords_map[rec_id],
            shot_coords=self.shot_to_coords_map[shot_id],
            samples=samples,
            sample_count=int(self.samp_num),  # kept here for easier debugging after gathering/collating
        )


def create_raw_trace_dataset(
    hdf5_path: typing.AnyStr,
    receiver_id_digit_count: int,  # could change from one site to the other
    first_break_field_name: str,   # name of the variable in the HDF5 where the fbp are located
    convert_to_fp16: bool = True,  # we don't need full fp32 precision by default
    convert_to_int16: bool = True,  # we don't need full int32/uint32 precision by default
    preload_trace_data: bool = False,  # might be useful on machines with lots of RAM
) -> RawTraceDataset:
    """Reads and returns the raw data contained in an HDF5 archive for first break picking.

    Also does a whole bunch of input validation in the process.

    Note: this parsing function assumes that all fields are the same across sites, and
    identical to the ones in the Halfmile_3D archive.
    """
    assert os.path.isfile(hdf5_path), f"cannot open hdf5 archive at path: {hdf5_path}"
    logger.info(f"parsing HDF5 data from: {hdf5_path}")
    dataset = RawTraceDataset(
        hdf5_path,
        receiver_id_digit_count=receiver_id_digit_count,
        first_break_field_name=first_break_field_name,
        convert_to_fp16=convert_to_fp16,
        convert_to_int16=convert_to_int16,
        preload_trace_data=preload_trace_data,
    )
    return dataset
