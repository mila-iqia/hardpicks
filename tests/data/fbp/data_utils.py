import os
from typing import List

import h5py
import numpy as np


def create_fake_traces(fake_dataset_params, seed):

    rng = np.random.default_rng(seed=seed)

    rec_count = fake_dataset_params["rec_count"]
    shot_count = fake_dataset_params["shot_count"]
    shot_peg_max = 10 ** fake_dataset_params["shot_peg_digit_count"]
    # shot ids/pegs are always unique (we're not looking at shot stations?), so just rng it up
    shot_pegs = rng.integers(
        shot_peg_max // 10, shot_peg_max, shot_count
    )
    fb_max_msec = (fake_dataset_params["samp_num"] - 1) * fake_dataset_params[
        "samp_rate"
    ]
    # for the trace parser, geometry should not matter, so x/y/z coords will be totally random
    rec_coords = rng.integers(1e6, 1e8, size=[rec_count, 3])
    shot_coords = rng.integers(1e6, 1e8, size=[shot_count, 3])
    # assign random receivers to random lines (again: we're not checking geometry/continuity here)
    rec_line_count = fake_dataset_params["rec_line_count"]
    rec_split_idxs = np.arange(1, rec_count - 1)
    rng.shuffle(rec_split_idxs)
    rec_split_idxs = (
        [0] + np.sort(rec_split_idxs[: (rec_line_count - 1)]).tolist() + [rec_count]
    )
    rec_id_max = 10 ** fake_dataset_params["rec_id_digit_count"]
    rec_ids = rng.integers(rec_id_max // 10, rec_id_max, rec_count)
    rec_peg_max = 10 ** fake_dataset_params["rec_peg_digit_count"]
    rec_pegs = [None] * rec_count  # will be initialized w/ proper line id in loop below
    rec_line_digits = (
        fake_dataset_params["rec_peg_digit_count"]
        - fake_dataset_params["rec_id_digit_count"]
    )
    rec_line_max = 10 ** rec_line_digits

    # now, create traces with random subsets of receivers, for all shots & lines
    traces = []
    for shot_idx in range(shot_count):
        for rec_line_idx in range(rec_line_count):
            curr_rec_idxs = np.arange(
                rec_split_idxs[rec_line_idx], rec_split_idxs[rec_line_idx + 1]
            )
            # Let's say... 50% chance to drop a couple of receivers (<5) on the line for each shot?
            # Let's make sure there's at least 3 receivers left on the line.
            if rng.random() < 0.5 and len(curr_rec_idxs) > 1:
                original_number_of_receivers_on_line = len(curr_rec_idxs)
                if original_number_of_receivers_on_line <= 3:
                    # don't drop any receivers
                    continue
                elif original_number_of_receivers_on_line == 4:
                    number_of_receivers_to_drop = 1
                else:
                    number_of_receivers_to_drop = rng.integers(
                        1, min(original_number_of_receivers_on_line - 3, 5)
                    )
                rng.shuffle(curr_rec_idxs)
                curr_rec_idxs = np.sort(curr_rec_idxs[number_of_receivers_to_drop:])
                assert len(curr_rec_idxs)
            for rec_idx in curr_rec_idxs:
                # if we have not assigned the proper full peg for this receiver, do it now
                if rec_pegs[rec_idx] is None:
                    line_id = (
                        rec_line_max // 10
                    ) + rec_line_idx  # just get a 10...-based line offset
                    offset_line_id = line_id * (
                        10 ** fake_dataset_params["rec_id_digit_count"]
                    )
                    rec_pegs[rec_idx] = offset_line_id + rec_ids[rec_idx]
                    assert rec_pegs[rec_idx] < rec_peg_max
                # here let's do... 10% chance to get a bad pick!
                if rng.random() < 0.1:
                    # both -1 and 0 are bad picks.
                    fb_msec = rng.choice([-1, 0])
                else:
                    # let's make sure we generate a random number that is far from zero
                    # to avoid rounding to zero in the bad pick detection code.
                    fb_msec = rng.integers(0.1 * fb_max_msec, fb_max_msec)
                # finally, generate a random array of samples, also with a 10% chance to be flat
                if rng.random() < 0.1:
                    samples = np.zeros((fake_dataset_params["samp_num"],))
                else:
                    samples = rng.random(
                        fake_dataset_params["samp_num"]
                    )
                traces.append(
                    dict(
                        shot_peg=shot_pegs[shot_idx],
                        shot_coords=shot_coords[shot_idx],
                        rec_peg=rec_pegs[rec_idx],
                        rec_coords=rec_coords[rec_idx],
                        fb_msec=fb_msec,
                        samples=samples,
                    )
                )

    return traces


def create_fake_first_break_picking_hdf5_file(
    traces: List[dict], fake_dataset_params: dict, output_hdf5_path: str
):
    """Function to write a fake hdf5 file to disk, for easy testing."""

    with h5py.File(output_hdf5_path, "w") as h5fd:
        # note: we'll fill only the bare minimum, it's not like the real deal at all!

        # start by filling in constants...
        samp_num_array = np.full(
            (len(traces),), fake_dataset_params["samp_num"], np.uint32
        )
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/SAMP_NUM", dtype=np.uint32, data=samp_num_array
        )
        samp_rate_array = np.full(
            (len(traces),), fake_dataset_params["samp_rate"] * 1000, np.uint32
        )
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/SAMP_RATE", dtype=np.uint32, data=samp_rate_array
        )
        coord_scale_array = np.full(
            (len(traces),), fake_dataset_params["coord_scale"], np.int32
        )
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/COORD_SCALE", dtype=np.uint32, data=coord_scale_array
        )
        ht_scale_array = np.full(
            (len(traces),), fake_dataset_params["ht_scale"], np.int32
        )
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/HT_SCALE", dtype=np.uint32, data=ht_scale_array
        )

        # fill in the receiver stuff from the traces...
        rec_peg_array = np.asarray([t["rec_peg"] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/REC_PEG", dtype=np.uint32, data=rec_peg_array
        )
        rec_coord_x_array = np.asarray([t["rec_coords"][0] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/REC_X", dtype=np.int32, data=rec_coord_x_array
        )
        rec_coord_y_array = np.asarray([t["rec_coords"][1] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/REC_Y", dtype=np.int32, data=rec_coord_y_array
        )
        rec_coord_z_array = np.asarray([t["rec_coords"][2] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/REC_HT", dtype=np.int32, data=rec_coord_z_array
        )

        # fill in the shot stuff from the traces...
        shot_peg_array = np.asarray([t["shot_peg"] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/SHOT_PEG", dtype=np.uint32, data=shot_peg_array
        )
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/SHOTID", dtype=np.uint32, data=shot_peg_array
        )
        shot_coord_x_array = np.asarray([t["shot_coords"][0] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/SOURCE_X", dtype=np.int32, data=shot_coord_x_array
        )
        shot_coord_y_array = np.asarray([t["shot_coords"][1] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/SOURCE_Y", dtype=np.int32, data=shot_coord_y_array
        )
        shot_coord_z_array = np.asarray([t["shot_coords"][2] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/SOURCE_HT", dtype=np.int32, data=shot_coord_z_array
        )

        # finally, fill in the trace samples and fb picks...
        samples_array = np.asarray([t["samples"] for t in traces])
        h5fd.create_dataset(
            "/TRACE_DATA/DEFAULT/data_array", dtype=np.float32, data=samples_array
        )
        fb_msec_array = np.asarray([t["fb_msec"] for t in traces])

        first_break_field_name = fake_dataset_params["first_break_field_name"]
        h5fd.create_dataset(
            f"/TRACE_DATA/DEFAULT/{first_break_field_name}", dtype=np.int32, data=fb_msec_array
        )


def create_fake_traces_and_hdf5_file(fake_dataset_params, seed, tmp_dir):
    fake_traces = create_fake_traces(fake_dataset_params, seed)
    output_hdf5_path = os.path.join(tmp_dir, f"fake_dataset_{seed}.hdf5")
    create_fake_first_break_picking_hdf5_file(
        fake_traces, fake_dataset_params, output_hdf5_path
    )
    return fake_traces, output_hdf5_path
