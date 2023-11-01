import typing
from pathlib import Path
from typing import Tuple

import scipy.stats as ss
import numpy as np
import pandas as pd
from tqdm import tqdm

from hardpicks.analysis.fbp.first_break_picking_seismic_data import (
    FirstBreakPickingSeismicData,
)
from hardpicks.data.fbp.constants import (
    DEAD_TRACE_AMPLITUDE_TOLERANCE,
)
from hardpicks.data.fbp.gather_cleaner import (
    ShotLineGatherCleaner,
)


def get_normalized_displacements(list_x, list_y):
    """Compute the normalized displacements in the plane."""
    list_dx = list_x[1:] - list_x[:-1]
    list_dy = list_y[1:] - list_y[:-1]
    norms = np.sqrt(list_dx**2 + list_dy**2)

    list_normalized_dx = np.concatenate([[np.NaN], list_dx / norms])
    list_normalized_dy = np.concatenate([[np.NaN], list_dy / norms])

    return list_normalized_dx, list_normalized_dy


def get_base_ax_index(
    number_of_plots: int, number_of_plots_per_row: int = 8
) -> Tuple[int, int]:
    """Compute the matplotlib ax base index number.

    The goal is to be able to iterate on the creation
    of figure axes without going over the number of
    allowed axes.
    """
    number_of_rows = np.ceil(number_of_plots / number_of_plots_per_row).astype(int)

    return number_of_rows, number_of_plots_per_row


def compute_basic_statistics(clean_dataset: ShotLineGatherCleaner) -> pd.DataFrame:
    """Compute basic statistics."""
    list_rows = []
    for idx in tqdm(range(len(clean_dataset)), desc="VALID ID"):
        datum = clean_dataset[idx]
        samples = datum["samples"]
        dead_trace_mask = np.abs(samples).mean(axis=1) < DEAD_TRACE_AMPLITUDE_TOLERANCE
        number_of_dead_traces = np.sum(dead_trace_mask)

        row = {
            "shot peg": datum["shot_id"],
            "line number": datum["rec_line_id"],
            "gather size": datum["trace_count"],
            "bad picks": sum(datum["bad_first_breaks_mask"]),
            "dead traces": number_of_dead_traces,
        }
        list_rows.append(row)

    return pd.DataFrame(list_rows)


def get_deduplicated_displacement_dataframe(measurements_df: pd.DataFrame):
    """Compute normalized measurements along receiver lines.

    Also remove duplicates for plotting purpose.
    """
    line_groups = measurements_df.groupby(by=["shot_peg", "line_number"])

    list_line_df = []
    for _, line_df in line_groups:
        del line_df["shot_peg"]
        line_df["dx"], line_df["dy"] = get_normalized_displacements(
            line_df["x"].values, line_df["y"].values
        )
        list_line_df.append(line_df)

    deduplicated_displacement_df = (
        pd.concat(list_line_df).reset_index(drop=True).drop_duplicates()
    )
    return deduplicated_displacement_df


def _is_injective(list_i: np.ndarray, list_j: np.ndarray) -> bool:
    """Check if each element of list_i is mapped to a single distinct element in list_j."""
    for i in np.unique(list_i):
        sub_j = list_j[list_i == i]
        if not len(np.unique(sub_j)) == 1:
            return False
    return True


def is_one_to_one(list_i: np.ndarray, list_j: np.ndarray) -> bool:
    """Check if the elements in list_i and list_j are in a one-to-one relationship."""
    return _is_injective(list_i, list_j) and _is_injective(list_j, list_i)


def get_data_from_site_dict(site_dict: dict):
    """Utility function to extract needed information."""
    site_name = site_dict["site_name"]

    data_file_path = Path(site_dict["processed_hdf5_path"])

    fbp_data = FirstBreakPickingSeismicData(
        data_file_path,
        receiver_id_digit_count=site_dict["receiver_id_digit_count"],
        first_break_pick_key=site_dict["first_break_field_name"],
    )
    return fbp_data, site_name, data_file_path


def compute_rms_amplitude(
    samples: typing.Sequence[float],
    pick_label: int,
    max_window_size: int,
    before: bool,
):
    """Returns the root mean square of the amplitude in a window near the picked location."""
    assert pick_label > 0 and pick_label < len(samples) - 1 and max_window_size > 0
    if before:
        win_min_idx, win_max_idx = max(pick_label - max_window_size, 0), pick_label
    else:
        win_min_idx, win_max_idx = pick_label, min(
            pick_label + max_window_size, len(samples)
        )
    return np.sqrt(np.power(samples[win_min_idx:win_max_idx], 2).mean())


def get_rms_amplitude_ratio(
    samples: np.array, first_break_index: int, max_window_size_samples: int
):
    """Compute the RMS amplitude ratio before and after first break pick."""
    rms_values = []
    for before in [True, False]:
        rms_value = compute_rms_amplitude(
            samples=samples,
            pick_label=first_break_index,
            max_window_size=max_window_size_samples,
            before=before,
        )
        rms_values.append(rms_value)

    rmsa_ratio = rms_values[0] / max(rms_values[1], 0.0001)
    return rmsa_ratio


def _rmse(list_errors: np.array):
    return np.sqrt(np.mean(list_errors**2))


def get_rmse_and_standard_deviation_bootstrapping(
    list_errors: np.array, number_of_resamples: int = 100, random_seed: int = 123
):
    """Get RMSE and an estimate of its standard deviation by bootstrapping.

    Args:
        list_errors (np.array): list of error values

    Returns:
        rmse (float): the root mean square error
        rmse_std (float): an estimate of the standard deviation of the RMSE
    """
    rng = np.random.default_rng(seed=random_seed)

    bootstrap_object = ss.bootstrap(
        (list_errors,),
        _rmse,
        n_resamples=number_of_resamples,
        vectorized=False,
        method="BCa",
        random_state=rng,
    )
    rmse = _rmse(list_errors)
    bootstrap_rmse_std = bootstrap_object.standard_error
    return rmse, bootstrap_rmse_std
