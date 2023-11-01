import logging

import matplotlib.pyplot as plt
from tqdm import tqdm

from hardpicks import (
    PLOT_STYLE_PATH,
    ANALYSIS_RESULTS_DIR,
)
from hardpicks.analysis.fbp.analysis_parser import get_fbp_data
from hardpicks.analysis.fbp.gather_plotting_utils import GatherAndContextPlotter
from hardpicks.analysis.logging_utils import (
    configure_logger_for_console_only,
)
from hardpicks.metrics.fbp.evaluator import FBPEvaluator

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
configure_logger_for_console_only(logger)

site_name = "Lalor"

pickle_path = (
    "/Users/bruno/monitoring/FBP/supplementary_experiments/"
    "supplement-predict/foldD/output_best-epoch=12-step=24894_valid.pkl"
)

list_interesting_gathers_and_trace_not_in_padding = [(250151, 141, 127), (234117, 137, 33)]
list_interesting_gathers_and_trace_in_padding = [(214158, 153, 138), (230119, 121, 11)]

if __name__ == "__main__":
    evaluator = FBPEvaluator.load(pickle_path)
    evaluator_df = evaluator._dataframe.reset_index(drop=True)

    logger.info("Reading seismic information...")
    fbp_data = get_fbp_data(site_name)

    plotter = GatherAndContextPlotter(fbp_data, evaluator_df)

    gathers = [list_interesting_gathers_and_trace_in_padding, list_interesting_gathers_and_trace_not_in_padding]
    list_around_ground_truth = [True, False]

    for list_interesting_gathers_and_trace, around_ground_truth in zip(gathers, list_around_ground_truth):
        for (shot_id, line_number, central_trace_index) in tqdm(list_interesting_gathers_and_trace):
            fig, output_file_name = plotter.plot_gather_and_minimal_context(shot_id, line_number,
                                                                            central_trace_index,
                                                                            around_ground_truth=around_ground_truth,
                                                                            show_title=False)
            image_path = ANALYSIS_RESULTS_DIR.joinpath(f"{site_name}_" + output_file_name)
            fig.savefig(image_path, bbox_inches="tight")
            plt.close(fig)
