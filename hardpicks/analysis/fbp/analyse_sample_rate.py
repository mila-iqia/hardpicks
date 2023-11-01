"""Extract sample rate and time grid analysis.

This script is meant to be executed. It iterates over all sites to extract
time-related information, such as the sample rate and the first break picks
alignment with the time grid.
"""
import numpy as np

from hardpicks import ANALYSIS_RESULTS_DIR
from hardpicks.analysis.fbp.analysis_utils import (
    get_data_from_site_dict,
)
from hardpicks.data.fbp.constants import (
    BAD_FIRST_BREAK_PICK_INDEX,
)
from hardpicks.data.fbp.site_info import get_site_info_array

output_path = ANALYSIS_RESULTS_DIR.joinpath("sample_rate_analysis.txt")

if __name__ == "__main__":

    site_info_array = get_site_info_array()
    with open(output_path, "w") as f:

        print(
            "# This text file presents a succinct analysis of the sample rate for each site",
            file=f,
        )
        print(
            "# and the relationship between first break picks and the implied time grid (ie time samples)",
            file=f,
        )
        for site_dict in site_info_array:

            fbp_data, site_name, data_file_path = get_data_from_site_dict(site_dict)

            bad_first_breaks_mask = (
                fbp_data.first_breaks_in_milliseconds <= BAD_FIRST_BREAK_PICK_INDEX
            )
            good_first_breaks_in_milliseconds = fbp_data.first_breaks_in_milliseconds[
                ~bad_first_breaks_mask
            ]

            fbp_indices_from_rounding = np.floor(
                good_first_breaks_in_milliseconds / fbp_data.sample_rate_milliseconds
            ).astype(int)

            good_first_breaks_in_milliseconds_from_rounding = (
                fbp_indices_from_rounding * fbp_data.sample_rate_milliseconds
            )

            absolute_errors_in_milliseconds = np.abs(
                good_first_breaks_in_milliseconds_from_rounding
                - good_first_breaks_in_milliseconds
            )
            maximum_absolute_error_in_milliseconds = np.max(
                absolute_errors_in_milliseconds
            )

            indices_found_mask = np.isin(
                good_first_breaks_in_milliseconds, fbp_data.time_in_milliseconds
            )
            maximum_absolute_error_when_found_in_milliseconds = np.max(
                absolute_errors_in_milliseconds[indices_found_mask]
            )

            first_breaks_absent_from_grid = good_first_breaks_in_milliseconds[
                ~indices_found_mask
            ]

            number_of_absents = len(first_breaks_absent_from_grid)

            number_of_good_fbp = (~bad_first_breaks_mask).sum()
            number_of_bad_fbp = bad_first_breaks_mask.sum()
            print(
                f"""
site {site_name}
-------------------------------
    - sample rate : {fbp_data.sample_rate_milliseconds} milliseconds
    - total number of fbp: {len(fbp_data.first_breaks_in_milliseconds)}
    - number of bad fbp: {number_of_bad_fbp}
    - number of good fbp: {number_of_good_fbp}
    - number of good fbp absent from time grid: {number_of_absents}  ({100*number_of_absents/number_of_good_fbp:2.1f} %)
    - maximum absolute error between good fbp and inferred fbp from rounded index:
            - all: {maximum_absolute_error_in_milliseconds} millisecond
            - for fbp found on grid: {maximum_absolute_error_when_found_in_milliseconds} millisecond
            """,
                file=f,
            )
