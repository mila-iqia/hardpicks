import os

import matplotlib.pylab as plt
import numpy as np

from hardpicks.analysis.fbp.report.path_constants import style_path, output_directory
from plot_lalor_shifted_picks_on_traces import get_trace_and_first_break
from hardpicks.analysis.fbp.first_break_picking_seismic_data import good_shot_peg_per_site, \
    FirstBreakPickingSeismicData
from hardpicks.data.fbp.site_info import get_site_info_by_name

plt.style.use(style_path)

image_path_template = os.path.join(str(output_directory),
                                   'Matagami_traces_{shot_peg}_{rec_peg}.png')

shot_peg = 31066
line_number = 1
list_rec_pegs = [1162, 1132, 1098]
xrange = (0, 500)
yrange = (-4.5, 4.5)

if __name__ == '__main__':

    site_name = 'Matagami'
    shot_peg_key = good_shot_peg_per_site[site_name]
    site_info = get_site_info_by_name(site_name)

    first_break_pick_key = 'SPARE1'
    fbp_data = FirstBreakPickingSeismicData(
        site_info["processed_hdf5_path"],
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        shot_peg_key=shot_peg_key,
        first_break_pick_key='SPARE1'
    )

    line_gather_indices = fbp_data.get_gather_indices(shot_peg, line_number)
    line_record_pegs = fbp_data.record_pegs[line_gather_indices]

    for rec_peg in list_rec_pegs:

        raw_trace, fbp = get_trace_and_first_break(fbp_data, shot_peg, line_number, rec_peg)

        line_gather_indices = fbp_data.get_gather_indices(shot_peg, line_number)
        line_record_pegs = fbp_data.record_pegs[line_gather_indices]

        peg_order = np.where(line_record_pegs == rec_peg)[0][0]

        if peg_order == 1:
            peg_postfix = f"({peg_order}" + "$^{st}$ peg on line)"
        else:
            peg_postfix = f"({peg_order}" + "$^{th}$ peg on line)"
        fig = plt.figure(figsize=(7.2, 4.45))
        fig.suptitle(f"{site_name} Site: Shot {shot_peg}, Line {line_number}, "
                     f"Peg {rec_peg} {peg_postfix}")
        ax = fig.add_subplot(111)

        ax.plot(fbp_data.time_in_milliseconds,
                raw_trace,
                color="blue",
                label='trace',
                lw=1,
                zorder=-1)

        amplitude = np.interp(fbp, fbp_data.time_in_milliseconds, raw_trace)
        ax.scatter([fbp], [amplitude], color="g", s=50, label="pick")

        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.set_ylabel('Amplitude (Arbitrary Units)')

        ax.legend(loc=0)
        ax.set_xlabel("Time (ms)")
        fig.tight_layout()

        image_path = image_path_template.format(shot_peg=shot_peg, rec_peg=rec_peg)
        fig.savefig(image_path)
        plt.close(fig)
