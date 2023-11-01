import logging
import os
from copy import copy
from shutil import copyfile

import h5py

from hardpicks.data.fbp.site_info import get_site_info_by_name

#  These pairs of swapped coordinates recorders were obtained by direct inspection

HALFMILE_BAD_RECORDER_PEGS_SWAP_PAIRS = [
    (10151384, 10032468),
    (10032476, 10151392),
    (10132051, 10112232),
]


def _get_bad_recorder_pegs_swap_dictionary(bar_recorder_pegs_swap_pairs):
    swap_dict = {}
    for peg1, peg2 in bar_recorder_pegs_swap_pairs:
        swap_dict[peg1] = peg2
        swap_dict[peg2] = peg1
    return swap_dict


def preprocess_halfmile_dataset(
    path_to_raw_dataset: str, path_to_output_processed_dataset: str
) -> None:
    """Create a processed copy of the Halfmile dataset.

    This function takes the path of the raw hdf5 file, makes a copy and fixes issues in the copy.

    A few pairs of receivers in the raw dataset have their identifiers (ie, the receiver peg)
    interchanged. That is to say that, for a few pairs of receivers, receiver A has the peg number of receiver B
    and vice versa. This function returns the correct identifiers to each receiver.
    """
    assert os.path.exists(path_to_raw_dataset), "input file does not exist"

    recorder_swap_dict = _get_bad_recorder_pegs_swap_dictionary(
        HALFMILE_BAD_RECORDER_PEGS_SWAP_PAIRS
    )

    logging.info(f"Copying {path_to_raw_dataset} to {path_to_output_processed_dataset}")
    copyfile(path_to_raw_dataset, path_to_output_processed_dataset)

    hdf5_file = h5py.File(path_to_output_processed_dataset, mode="r+")

    rec_pegs = hdf5_file["/TRACE_DATA/DEFAULT/REC_PEG"]

    raw_rec_pegs_values = rec_pegs[:].flatten()

    processed_rec_pegs_values = copy(raw_rec_pegs_values)

    logging.info("Fixing receiver pegs in copied file")
    for index, record_peg in enumerate(raw_rec_pegs_values):
        if record_peg in recorder_swap_dict:
            correct_rec_peg = recorder_swap_dict[record_peg]
            logging.debug(
                f"Substituting peg {record_peg} with peg {correct_rec_peg}"
            )
            processed_rec_pegs_values[index] = correct_rec_peg

    rec_pegs[...] = processed_rec_pegs_values.reshape(-1, 1)
    hdf5_file.close()


if __name__ == "__main__":
    halfmile_site_info = get_site_info_by_name("Halfmile")
    preprocess_halfmile_dataset(
        halfmile_site_info["raw_hdf5_path"], halfmile_site_info["processed_hdf5_path"]
    )
