import os
from typing import List

import h5py
import numpy as np
from tqdm import tqdm

from hardpicks.data.fbp.site_info import get_site_info_by_name
from hardpicks.data.fbp.trace_parser import BASE_EXPECTED_HDF5_FIELDS

#  These receiver pegs are not along the receiver lines of the Matagami site. They
#  were obtained by inspection.
_BAD_RECEIVER_PEGS = np.array(
    [
        1001,
        1002,
        33007,
        33008,
        33009,
        33010,
        33011,
        33012,
        33013,
        33014,
        33015,
        33016,
        33017,
        33019,
        33020,
        33021,
        33023,
        33025,
        33027,
        33029,
        33030,
        33031,
        33032,
        33033,
        33034,
        33035,
        33036,
    ]
)


def get_good_receiver_indices(
    all_receiver_pegs: np.ndarray, bad_receiver_pegs: np.ndarray
) -> np.ndarray:
    """Find the indices of the bad receiver pegs in the array of all pegs."""
    mask = np.isin(all_receiver_pegs, bad_receiver_pegs)
    good_receiver_indices = np.arange(len(all_receiver_pegs))[~mask]
    return good_receiver_indices


def preprocess_matagami_dataset(
    path_to_raw_dataset: str,
    path_to_output_processed_dataset: str,
    list_hdf5_fields: List[str],
    bad_receiver_pegs: np.array = _BAD_RECEIVER_PEGS,
) -> None:
    """Create a processed copy of the Matagami dataset.

    This function takes the path of the raw hdf5 file and creates a new
    file in which it writes the corrected data. This approach is used
    instead of modifying a copy because it is difficult to modify
    the dimensions of hdf5 arrays in place.
    """
    assert os.path.exists(path_to_raw_dataset), "input file does not exist"

    raw_hdf5_file = h5py.File(path_to_raw_dataset, mode="r")
    processed_hdf5_file = h5py.File(path_to_output_processed_dataset, mode="x")

    all_receiver_pegs = raw_hdf5_file["/TRACE_DATA/DEFAULT/REC_PEG"][:].flatten()
    good_receiver_indices = get_good_receiver_indices(
        all_receiver_pegs, bad_receiver_pegs
    )

    for field in tqdm(list_hdf5_fields, desc="FIELD"):

        field_path = f"/TRACE_DATA/DEFAULT/{field}"
        all_values = raw_hdf5_file[field_path][:]
        processed_hdf5_file.create_dataset(field_path, data=all_values[good_receiver_indices], track_times=False)

    raw_hdf5_file.close()
    processed_hdf5_file.close()


if __name__ == "__main__":
    matagami_site_info = get_site_info_by_name("Matagami")
    # to keep the original preprocessed hash, we need SPARE1 in its original place...
    hdf5_fields = BASE_EXPECTED_HDF5_FIELDS.copy()
    hdf5_fields.insert(-1, matagami_site_info["first_break_field_name"])  # before-last
    preprocess_matagami_dataset(
        matagami_site_info["raw_hdf5_path"],
        matagami_site_info["processed_hdf5_path"],
        list_hdf5_fields=hdf5_fields,
    )
