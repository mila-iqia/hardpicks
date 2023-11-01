"""Description of analysis script.

This simple script establishes the relationship between the SHOT_PEG and SHOTID fields for the different
sites. Execution should print to console whether these arrays are the same, one-to-one or completely different.
"""
import numpy as np

from hardpicks.analysis.fbp.analysis_utils import is_one_to_one, get_data_from_site_dict
from hardpicks.data.fbp.site_info import get_site_info_array

if __name__ == "__main__":

    site_info_array = get_site_info_array()
    for site_dict in site_info_array:

        fbp_data, site_name, data_file_path = get_data_from_site_dict(site_dict)

        shot_ids = fbp_data.get_one_dimensional_dataset("SHOTID")
        shot_pegs = fbp_data.get_one_dimensional_dataset("SHOT_PEG")

        one_to_one = is_one_to_one(shot_ids, shot_pegs)

        if np.linalg.norm(shot_ids - shot_pegs) < 1e-6:
            print(f"{site_name}: SHOTID and SHOT_PEG are the same")
        elif one_to_one:
            print(f"{site_name}: SHOTID and SHOT_PEG are one to one")
        else:
            print(f"{site_name}: SHOTID and SHOT_PEG are NOT one to one")
            print(f"   number of unique SHOTIDS: {len(np.unique(shot_ids))}")
            print(f"   number of unique SHOT_PEGS: {len(np.unique(shot_pegs))}")
