import matplotlib.pylab as plt
import numpy as np

from hardpicks.analysis.fbp.report.path_constants import style_path, output_directory
from hardpicks.analysis.fbp.first_break_picking_seismic_data import good_shot_peg_per_site, \
    FirstBreakPickingSeismicData
from hardpicks.data.fbp.site_info import get_site_info_by_name

plt.style.use(style_path)

image_path = output_directory.joinpath('Lalor_shifted_picks.png')


def get_trace_and_first_break(first_break_picking_seismic_data, shot_peg, line_number, rec_peg):
    """Convenience method to get the trace and pick for a specific shot, line and record peg."""
    line_gather_indices = first_break_picking_seismic_data.get_gather_indices(shot_peg, line_number)
    shot_record_pegs = first_break_picking_seismic_data.record_pegs[line_gather_indices]

    idx = np.where(shot_record_pegs == rec_peg)[0][0]
    trace_index = line_gather_indices[idx]

    raw_trace = first_break_picking_seismic_data.raw_traces[trace_index, :]
    first_break_pick = first_break_picking_seismic_data.first_breaks_in_milliseconds[trace_index]
    return raw_trace, first_break_pick


shot_peg = 230152
line_number = 145
list_rec_pegs = [145204, 145224, 145244]

if __name__ == '__main__':

    site_name = 'Lalor'
    shot_peg_key = good_shot_peg_per_site[site_name]
    site_info = get_site_info_by_name(site_name)

    first_break_pick_key = 'SPARE1'
    fbp_data = FirstBreakPickingSeismicData(
        site_info["processed_hdf5_path"],
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        shot_peg_key=shot_peg_key,
        first_break_pick_key='SPARE1'
    )
    corrected_fbp_data = FirstBreakPickingSeismicData(
        site_info["processed_hdf5_path"],
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        shot_peg_key=shot_peg_key,
        first_break_pick_key='SPARE2'
    )

    fig = plt.figure(figsize=(7.2, 4.45))
    fig.suptitle(f"{site_name} Site: Shot {shot_peg} and Receiver Line {line_number}")

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    list_ax = [ax1, ax2, ax3]

    list_xranges = [(0, 50), (75, 125), (150, 200)]
    list_yranges = [(-100, 100), (-10, 10), (-1.5, 1.5)]

    # Add inset to clearly show that the shifted annotation is at minimum
    ax_inset = ax1.inset_axes([0.15, 0.10, 0.25, 0.25])
    for spine in ax_inset.spines.values():
        spine.set_visible(True)

    first = True
    for ax, rec_peg, xrange, yrange in zip(list_ax, list_rec_pegs, list_xranges, list_yranges):
        raw_trace, fbp = get_trace_and_first_break(fbp_data, shot_peg, line_number, rec_peg)
        _, corrected_fbp = get_trace_and_first_break(corrected_fbp_data, shot_peg, line_number, rec_peg)
        ax.set_title(f"Peg {rec_peg}", loc='center')

        amplitude = np.interp(fbp, fbp_data.time_in_milliseconds, raw_trace)
        corrected_amplitude = np.interp(corrected_fbp, corrected_fbp_data.time_in_milliseconds, raw_trace)

        if first:
            list_sub_axes = [ax, ax_inset]
            first = False
        else:
            list_sub_axes = [ax]

        for sub_ax in list_sub_axes:
            sub_ax.plot(fbp_data.time_in_milliseconds, raw_trace, color="blue", label='trace', zorder=-1)
            sub_ax.scatter([fbp], [amplitude], color="r", label="original pick")
            sub_ax.scatter([corrected_fbp], [corrected_amplitude], color="g", label="shifted pick")

        ax.set_xlabel("Time (ms)")
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)

    ax1.legend(loc='upper left')
    ax1.set_ylabel('Amplitude (Arbitrary Units)')

    x1, x2, y1, y2 = 35, 39, -30, -22
    ax_inset.set_xlim(x1, x2)
    ax_inset.set_ylim(y1, y2)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_xticklabels('')
    ax_inset.set_yticklabels('')
    ax1.indicate_inset_zoom(ax_inset, edgecolor="black")

    fig.tight_layout()

    fig.savefig(image_path)
    plt.close(fig)
