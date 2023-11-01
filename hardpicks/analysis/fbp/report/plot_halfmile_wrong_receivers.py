import matplotlib.pylab as plt
import numpy as np


from hardpicks.analysis.fbp.report.path_constants import style_path, output_directory
from hardpicks.analysis.fbp.first_break_picking_seismic_data import good_shot_peg_per_site, \
    FirstBreakPickingSeismicData
from hardpicks.data.fbp.receiver_location_corrections import \
    HALFMILE_BAD_RECORDER_PEGS_SWAP_PAIRS
from hardpicks.data.fbp.site_info import get_site_info_by_name

plt.style.use(style_path)

image_path = output_directory.joinpath('halfmile_wrong_receiver_pegs.png')

if __name__ == "__main__":

    site_name = 'Halfmile'

    suspicious_receiver_lines = [1003, 1011, 1013, 1015]
    list_colors = ['coral', 'dodgerblue', 'orange', 'navy']

    shot_peg_key = good_shot_peg_per_site[site_name]
    site_info = get_site_info_by_name(site_name)

    fbp_data = FirstBreakPickingSeismicData(
        site_info["raw_hdf5_path"],
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        shot_peg_key=shot_peg_key,
    )

    recorder_df = fbp_data.get_recorder_dataframe()
    origin = recorder_df.x.min(), recorder_df.y.min()

    line_groups = recorder_df.groupby(by="line_number")

    fig = plt.figure(figsize=(7.2, 4.45))

    fig.suptitle(f"{site_name} Site: Incorrect Receiver Pegs")
    ax = fig.add_subplot(111)
    ax.scatter(
        recorder_df.x - origin[0],
        recorder_df.y - origin[1],
        color="grey",
        alpha=0.15,
        s=10,
        label="all geophones",
    )

    bad_receiver_pegs = np.concatenate(HALFMILE_BAD_RECORDER_PEGS_SWAP_PAIRS)

    for mask_type, size in zip(['good', 'bad'], [10, 20]):
        # Loop twice so that the bad data is on top of the good data on the plot.
        for line_number, color in zip(suspicious_receiver_lines, list_colors):
            group_df = line_groups.get_group(line_number)

            bad_peg_mask = group_df.index.isin(bad_receiver_pegs)
            good_peg_mask = ~bad_peg_mask
            if mask_type == 'good':
                peg_mask = ~bad_peg_mask
                label = f"receiver line {line_number}"
            else:
                peg_mask = bad_peg_mask
                label = "__nolabel__"

            ax.scatter(group_df[peg_mask].x - origin[0],
                       group_df[peg_mask].y - origin[1],
                       color=color,
                       s=size,
                       label=label)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc=0)

    fig.savefig(image_path)
    plt.close(fig)
