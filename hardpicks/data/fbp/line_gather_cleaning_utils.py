"""Utility functions.

This module contains useful utility methods to compute
fits of source-receiver distance versus first break picks.
"""
import logging
import typing

import numpy as np
from tqdm import tqdm

if typing.TYPE_CHECKING:
    import hardpicks.data.fbp.gather_parser as gather_parser

logger = logging.getLogger(__name__)


def fit_distance_with_time(offset_distances: np.ndarray, times: np.ndarray):
    """Creates a linear fit of the time vs. distance relationship.

    The linear fit is performed using the equation:

    x = v0 * (t - t0)

    where x is distance and t is time. In the fit, v0 is interpreted as the "real" velocity
    of the wave propagation, and t0 is the time offset between start of measurement and shot.

    Args:
        offset_distances: distances between source and receivers
        times: first break picking times corresponding to each distances

    Returns:
        v0: the fitted velocity
        t0: the offset time.
    """
    # convert to float64 in case the inputs are in lower precision. numpy.linalg cannot operate
    # on float16 data.
    v0, minus_v0t0 = np.polyfit(times.astype(np.float64), offset_distances.astype(np.float64), deg=1)

    t0 = -minus_v0t0 / v0

    return v0, t0


def unpack_data_dict(data_dict):
    """Extract useful info from dict.

    Args:
        data_dict: item from a site Dataset

    Returns:
        offset_distances: distance between source and receiver
        first_break_times_in_milliseconds: first break pick times
        bad_mask: mask identifying the bad first break picks
    """
    first_break_times_in_milliseconds = data_dict["first_break_timestamps"]
    offset_distances = data_dict["offset_distances"][:, 0]
    bad_mask = data_dict["bad_first_breaks_mask"]
    return offset_distances, first_break_times_in_milliseconds, bad_mask


def get_good_times_and_distances(
    shot_line_gather_dataset: "gather_parser.ShotLineGatherDatasetBase",
):
    """Compute all times and distances arrays.

     Args:
        shot_line_gather_dataset: a shot line gather parser (or wrapped object)

    Returns:
        list_distances: a numpy array that contains source-receiver distances for valid traces
        list_times: a numpy array that contains first break times for valid traces
    """
    gather_ids = np.arange(len(shot_line_gather_dataset))

    list_good_distances = []
    list_good_times = []
    for idx in tqdm(gather_ids, "Extract distance and time for linear fit"):
        try:
            datum = shot_line_gather_dataset.get_meta_gather(idx)
        except Exception as e:
            logger.error(f"Cannot load idx {idx}. Error: {e.args[0]}")
            continue

        distances, first_break_times_in_milliseconds, bad_mask = unpack_data_dict(datum)
        list_good_distances.append(distances[~bad_mask])
        list_good_times.append(first_break_times_in_milliseconds[~bad_mask])

    return np.concatenate(list_good_distances), np.concatenate(list_good_times)


def get_global_linear_fit_and_all_errors(
    shot_line_gather_dataset: "gather_parser.ShotLineGatherDatasetBase",
):
    """Create linear fit to all good distance vs time data and compute absolute errors.

    This function extracts velocity and time offset fit parameters and fit errors
    for all the good entries in the dataset.

    Args:
        shot_line_gather_dataset: a shot line gather parser (or wrapped object)

    Returns:
        v0: the globally fitted velocity parameter
        t0: the globally fitted time offset parameter
        all_absolute_errors: the concatenation of all trace absolute errors vs. the linear fit.
    """
    all_good_distances, all_good_times = get_good_times_and_distances(shot_line_gather_dataset)

    # Use float64 to avoid overflow.
    all_good_distances_in_float64 = all_good_distances.astype(np.float64)
    all_good_times_in_float64 = all_good_times.astype(np.float64)

    v0, t0 = fit_distance_with_time(all_good_distances_in_float64, all_good_times_in_float64)

    all_fitted_distances_in_float64 = v0 * (all_good_times_in_float64 - t0)

    all_absolute_errors = np.abs(all_good_distances_in_float64 - all_fitted_distances_in_float64)

    return v0, t0, all_absolute_errors


def get_fitted_distances(v0, t0, times):
    """Compute fitted distances.

    Args:
        v0: fitted velocity
        t0: fitted time offset
        times: numpy array containing times

    Returns:
        fitted_distances: numpy array containing the distances according to the linear fit
    """
    return v0 * (times - t0)
