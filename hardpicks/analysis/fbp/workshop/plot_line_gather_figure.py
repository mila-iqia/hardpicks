import logging
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from hardpicks import ANALYSIS_RESULTS_DIR

from hardpicks.data.fbp.gather_parser import create_shot_line_gather_dataset
from hardpicks.data.fbp.site_info import get_site_info_by_name
from hardpicks.data.fbp.gather_preprocess import ShotLineGatherPreprocessor

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    gather_top_region = (100, 700)
    gather_resize_height = 220
    vertical_line_width = 1

    target_site_name = "Halfmile"
    target_shot_id = 20061620
    target_gather_line_idxs = slice(1, 4)

    plot_gather_idx = 2
    plot_gather_sample_range = (20, 300)

    site_info = get_site_info_by_name(target_site_name)

    dataset = create_shot_line_gather_dataset(
        hdf5_path=site_info["processed_hdf5_path"],
        site_name=target_site_name,
        first_break_field_name='SPARE1',
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        cache_trace_metadata=True,
        provide_offset_dists=True,
    )

    dataset_wrapped = ShotLineGatherPreprocessor(
        dataset,
        normalize_samples=True,
    )

    for plot_trace_idx in range(100, 180, 2):

        shot_id_to_gather_ids_map = {}
        for gid, tids in dataset.gather_to_trace_map.items():
            sids = [dataset.trace_to_shot_map[t] for t in tids]
            assert len(np.unique(sids)) == 1
            sid = sids[-1]
            if sid not in shot_id_to_gather_ids_map:
                shot_id_to_gather_ids_map[sid] = []
            shot_id_to_gather_ids_map[sid].append(gid)

        gather_ids = shot_id_to_gather_ids_map[target_shot_id][target_gather_line_idxs]

        gathers, gather_widths, gather_fb_idxs, gather_fb_tstamps = [], [], [], []
        for gather_id in gather_ids:
            gather_data = dataset_wrapped[gather_id]
            samples = gather_data["samples"][:, gather_top_region[0]:gather_top_region[1]].T
            gathers.append(samples)
            gather_fb_idxs.append((gather_data["first_break_labels"] - gather_top_region[0]).astype(np.float32))
            gather_fb_tstamps.append(gather_data["first_break_timestamps"].astype(np.float32))
            gather_widths.append(samples.shape[1])
        tot_vline_width = (len(gathers) - 1) * vertical_line_width
        display = np.zeros(
            (gather_resize_height, sum(gather_widths) + tot_vline_width),
            np.float32,
        )
        curr_x_offset = 0
        for gather, gather_width, fb_idxs in zip(gathers, gather_widths, gather_fb_idxs):
            fb_idxs *= gather_resize_height / gather.shape[0]
            gather = cv.resize(
                gather,
                dsize=(gather_width, gather_resize_height),
                interpolation=cv.INTER_LINEAR,
            )
            display[:, curr_x_offset:(curr_x_offset + gather.shape[1])] = gather
            curr_x_offset += gather.shape[1] + vertical_line_width
        display = cv.normalize(display, display, 0, 255, norm_type=cv.NORM_MINMAX)
        display = cv.cvtColor(display.astype(np.uint8), cv.COLOR_GRAY2BGR)
        curr_x_offset = 0
        for gather_idx, gather_width in enumerate(gather_widths):
            for col_idx, fb_idx in enumerate(gather_fb_idxs[gather_idx]):
                display = cv.circle(
                    display,
                    (col_idx + curr_x_offset, int(round(fb_idx))),
                    1,
                    (255, 0, 0),
                    -1,
                )
            if gather_idx == plot_gather_idx:
                display = cv.line(
                    display,
                    (plot_trace_idx + curr_x_offset, 0),
                    (plot_trace_idx + curr_x_offset, gather_resize_height),
                    (0, 0, 255),
                    1,
                )
            if gather_idx == len(gathers) - 1:
                break
            curr_x_offset += gather_width
            display = cv.line(
                display,
                (curr_x_offset + vertical_line_width // 2, 0),
                (curr_x_offset + vertical_line_width // 2, gather_resize_height),
                (0, 255, 0),
                thickness=vertical_line_width,
            )
            curr_x_offset += vertical_line_width
        display = cv.resize(display, dsize=(-1, -1), fx=4, fy=4, interpolation=cv.INTER_NEAREST)
        cv.imshow("display", display)

        fig = plt.figure(figsize=(3.5, 2.5), dpi=320)
        ax = fig.add_subplot(111)
        time_range_ms = (
            np.arange(plot_gather_sample_range[1] - plot_gather_sample_range[0])
            + gather_top_region[0] + plot_gather_sample_range[0]
        ) * dataset.samp_rate / 1000
        target_trace = gathers[plot_gather_idx][:, plot_trace_idx]
        target_samples = target_trace[plot_gather_sample_range[0]:plot_gather_sample_range[1]]
        ax.plot(time_range_ms, target_samples, "k-")
        target_fb_tstamp = gather_fb_tstamps[plot_gather_idx][plot_trace_idx]
        ax.axvline(x=target_fb_tstamp, color="b", linestyle="--")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (Arbitrary Units)")
        fig.tight_layout()
        image_path = os.path.join(ANALYSIS_RESULTS_DIR, "mini-trace-plot.pdf")
        fig.show()
        fig.savefig(image_path)

        cv.waitKey(0)
