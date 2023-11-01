import logging

import matplotlib.pyplot as plt
from tqdm import tqdm

from hardpicks import (
    PLOT_STYLE_PATH, ANALYSIS_RESULTS_DIR,
)
from hardpicks.analysis.fbp.analysis_parser import get_fbp_data
from hardpicks.analysis.fbp.gather_plotting_utils import GatherAndContextPlotter
from hardpicks.analysis.logging_utils import (
    configure_logger_for_console_only,
)

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
configure_logger_for_console_only(logger)

site_name = "Lalor"

output_dir = ANALYSIS_RESULTS_DIR / "lalor_pick_quality"
output_dir.mkdir(exist_ok=True)


# these various examples were selected by inspection by looking at some of our other reports.
bad_list_interesting_gathers_and_trace = [(234160, 141, 121), (234174, 129, 149),
                                          (230148, 153, 131), (242132, 153, 106),
                                          (234156, 145, 72), (218121, 137, 135),
                                          (230130, 101, 116), (222127, 137, 99),
                                          (230161, 117, 142), (234169, 125, 157),
                                          (230166, 149, 101)]

good_list_interesting_gathers_and_trace = [(222140, 121, 70), (230139, 145, 44), (222123, 113, 18)]

if __name__ == "__main__":
    logger.info("Reading seismic information...")
    fbp_data = get_fbp_data(site_name)

    plotter = GatherAndContextPlotter(fbp_data)
    plotter.show_original_ground_truth_picks(flag=True)
    plotter.show_ground_truth_picks(flag=True)
    plotter.show_multitrace_context(around_ground_truth=True, around_predictions=False)
    plotter.show_central_trace_zoom_in(around_ground_truth=True, around_predictions=False)

    interesting = [bad_list_interesting_gathers_and_trace, good_list_interesting_gathers_and_trace]
    labels = ['AMBIGUOUS', 'GOOD']
    for list_interesting_gathers_and_trace, label in zip(interesting, labels):
        for (shot_id, line_number, central_trace_index) in tqdm(
            list_interesting_gathers_and_trace
        ):
            output_filename = f"{site_name}_{label}_line_gather_pick_quality" \
                              f"_shot_{shot_id}_line_{line_number}_trace_{central_trace_index}.png"
            fig_and_ax_dict = plotter.generate_plot(shot_id, line_number, central_trace_index, show_title=False)
            fig = fig_and_ax_dict['fig']
            image_path = output_dir / output_filename
            fig.savefig(image_path)
            plt.close(fig)
