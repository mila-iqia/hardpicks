import matplotlib.pyplot as plt
import numpy as np


from hardpicks.analysis.fbp.report.path_constants import style_path, output_directory
from hardpicks import FBP_DATA_DIR
from hardpicks.analysis.fbp.analyse_pick_shifts.create_multi_gathers_figures import \
    get_multi_gathers_figure
from hardpicks.analysis.fbp.first_break_picking_seismic_data import (
    FirstBreakPickingSeismicData,
    good_shot_peg_per_site,
)

plt.style.use(style_path)

image_path = output_directory.joinpath('brunswick_multi_gather.png')

bad_shot_id = 11108

# Old Brunswick info
site_name = 'Brunswick'
hdf5_path = FBP_DATA_DIR.joinpath("Brunswick_3D/Brunswick_orig_1500ms.hdf5")
receiver_id_digit_count = 3

if __name__ == "__main__":

    shot_peg_key = good_shot_peg_per_site[site_name]

    fbp_data = FirstBreakPickingSeismicData(
        path_to_hdf5_file=hdf5_path,
        receiver_id_digit_count=receiver_id_digit_count,
        shot_peg_key=shot_peg_key,
    )

    list_pairs = list(zip(fbp_data.shot_pegs, fbp_data.record_line_numbers))

    list_shot_pegs = []
    list_line_numbers = []
    set_of_known_pairs = set()
    for pair in list_pairs:
        if pair not in set_of_known_pairs:
            set_of_known_pairs.add(pair)
            shot_peg, line_number = pair
            list_shot_pegs.append(shot_peg)
            list_line_numbers.append(line_number)

    list_shot_pegs = np.array(list_shot_pegs)
    list_line_numbers = np.array(list_line_numbers)

    shot_mask = list_shot_pegs == bad_shot_id

    fig = get_multi_gathers_figure(
        fbp_data, list_shot_pegs[shot_mask], list_line_numbers[shot_mask]
    )

    fig.suptitle(f"{site_name} Site: Shot Gather for Shot {bad_shot_id}")

    fig.savefig(image_path)
    plt.close(fig)
