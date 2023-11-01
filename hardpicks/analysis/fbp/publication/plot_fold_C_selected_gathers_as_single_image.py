import logging

import matplotlib.pyplot as plt

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

site_name = "Brunswick"
# Note that reading this pickle requires Pandas 1.5.3; it is incompatible with pandas >= 2.0
pickle_path = "/Users/bruno/monitoring/FBP/supplementary_experiments/supplement-predict/" \
              "foldC/output_best-epoch=14-step=35654_valid.pkl"

interesting_gathers = [(151111, 541), (271023, 141), (91002, 161), (171001, 101)]

if __name__ == "__main__":
    evaluator = FBPEvaluator.load(pickle_path)
    evaluator_df = evaluator._dataframe.reset_index(drop=True)

    logger.info("Reading seismic information...")
    fbp_data = get_fbp_data(site_name)

    plotter = GatherAndContextPlotter(fbp_data, evaluator_df)

    show_legend = True
    # Generate each image separately
    for (shot_gather, line_number) in interesting_gathers:
        fig, output_file_name = plotter.generate_esthetic_gather_plot(shot_peg=shot_gather,
                                                                      line_number=line_number,
                                                                      show_legend=show_legend)
        show_legend = False
        image_path = ANALYSIS_RESULTS_DIR / f"{site_name}_{output_file_name}"
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)

    fig = plotter.generate_multi_line_gather_plot(interesting_gathers)

    image_path = ANALYSIS_RESULTS_DIR / f"{site_name}_gather_with_airwave_error_examples.png"
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
