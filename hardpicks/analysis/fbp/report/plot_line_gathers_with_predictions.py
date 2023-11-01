import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from hardpicks.analysis.fbp.report.path_constants import (
    style_path,
    pickles_directory,
    output_directory,
)
from hardpicks.analysis.fbp.velocity_analysis.line_gather_plotter import (
    ReportSiteLineGatherPlotter,
    PickProperties,
)
from hardpicks.analysis.logging_utils import (
    setup_analysis_logger,
)

from hardpicks.data.fbp.gather_parser import (
    create_shot_line_gather_dataset,
)
from hardpicks.data.fbp.site_info import get_site_info_by_name
from hardpicks.metrics.fbp.evaluator import FBPEvaluator


class ReportSiteLineGatherWithTracePlotter(ReportSiteLineGatherPlotter):
    """Generate line gather plots that is report quality, with a second ax showing a trace."""

    predict_color = "orange"
    predict_size = 10

    def _plot_predictions(
        self, predictions_in_milliseconds, prediction_pick_properties, ax
    ):
        number_of_traces = len(predictions_in_milliseconds)
        list_trace_index = np.arange(number_of_traces)

        ax.scatter(
            list_trace_index[prediction_pick_properties.mask],
            predictions_in_milliseconds[prediction_pick_properties.mask],
            s=prediction_pick_properties.size,
            marker=prediction_pick_properties.symbol,
            facecolors=prediction_pick_properties.color,
            edgecolor=prediction_pick_properties.color,
            label=prediction_pick_properties.label,
        )

    def _plot_trace(self, datum, list_predictions_in_milliseconds, trace_index, ax):

        trace = datum["samples"][trace_index]

        ground_truth_fbp = datum["first_break_timestamps"][trace_index]
        predicted_fbp = list_predictions_in_milliseconds[trace_index]

        number_of_samples = len(trace)
        times_in_milliseconds = self._sampling_time_in_milliseconds * np.arange(
            number_of_samples
        )

        ax.plot(times_in_milliseconds, trace, "b-", lw=1, label="raw trace", zorder=-1)

        ground_truth_amplitude = np.interp(
            ground_truth_fbp, times_in_milliseconds, trace
        )
        predicted_amplitude = np.interp(predicted_fbp, times_in_milliseconds, trace)

        good_pick_properties, _ = self._get_pick_properties(datum)

        ax.scatter(
            [ground_truth_fbp],
            [ground_truth_amplitude],
            s=good_pick_properties.size,
            color=good_pick_properties.color,
            label="ground truth pick",
        )

        ax.scatter(
            [predicted_fbp],
            [predicted_amplitude],
            s=self.predict_size,
            color=self.predict_color,
            label="predicted pick",
        )

        ax.set_xlim(0, times_in_milliseconds[-1])

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (Arb. Units)")

    def create_figure(self, data_dict, predictions_in_milliseconds, trace_index):
        """Plot a line gather with annotated first break picks.

        Args:
            data_dict: single sample from a ShotLineGatherDataset

        Returns:
            fig: matplotlib figure with one ax
            ax: matplotlib ax with the plot.
        """
        fig = plt.figure(figsize=(7.2, 4.45))

        gs = fig.add_gridspec(nrows=3, ncols=1)
        ax1 = fig.add_subplot(gs[:-1, :])
        ax2 = fig.add_subplot(gs[-1, :])

        fig.suptitle(
            f"{self._site_name} Site: Shot {data_dict['shot_id']}, Receiver Line {data_dict['rec_line_id']}"
        )

        self._plot_line_gather(data_dict, fig, ax1, plot_colorbar=False)
        self._plot_picks(data_dict, ax1)

        good_prediction_mask = ~np.isnan(predictions_in_milliseconds)
        prediction_pick_properties = PickProperties(
            mask=good_prediction_mask,
            size=self.predict_size,
            symbol="o",
            color=self.predict_color,
            label="predicted picks",
        )

        self._plot_predictions(
            predictions_in_milliseconds, prediction_pick_properties, ax1
        )
        ax1.legend(loc=3)

        self._plot_trace(datum, predictions_in_milliseconds, trace_index, ax2)
        ax2.set_title(f"Trace Index {trace_index}")

        fig.tight_layout()

        return fig, ax1, ax2


logger = logging.getLogger(__name__)
setup_analysis_logger()

plt.style.use(style_path)

# plots can be fiddly. It's faster to have the needed data on hand.
generate_pickles = True

pickle_path_template = os.path.join(
    str(pickles_directory), "{site_name}_{shot_id}_{line_id}_datum.pkl"
)

image_path_template = os.path.join(
    str(output_directory),
    "{site_name}_gather_and_trace_{shot_id}_{line_id}_{rec_index}.png",
)


site_name = "Sudbury"
list_parameters = [
    dict(shot_id=505, line_id=22, rec_index=46, xmin=0, xmax=1000.0),
    dict(shot_id=823, line_id=16, rec_index=176, xmin=0, xmax=1000.0),
    dict(shot_id=954, line_id=26, rec_index=47, xmin=0, xmax=1000.0),
    dict(shot_id=282, line_id=24, rec_index=58, xmin=230, xmax=250),
    dict(shot_id=378, line_id=23, rec_index=53, xmin=250, xmax=290),
    dict(shot_id=904, line_id=24, rec_index=70, xmin=250, xmax=275),
    dict(shot_id=503, line_id=26, rec_index=5, xmin=160, xmax=200),
]

base_mlrun_folder_path = "/Users/bruno/monitoring/orion/foldB/mlruns/5/9d38db56316c4b4989fddbb7369c8ca1/artifacts/"
data_dump_path = os.path.join(
    base_mlrun_folder_path, "data_dumps/output_best-epoch=11-step=38243_valid.pkl"
)


if __name__ == "__main__":

    evaluator = FBPEvaluator.load(data_dump_path)

    evaluation_df = evaluator._dataframe

    site_info = get_site_info_by_name(site_name)
    shot_line_gather_dataset = create_shot_line_gather_dataset(
        hdf5_path=site_info["processed_hdf5_path"],
        site_name=site_name,
        first_break_field_name="SPARE1",
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        convert_to_fp16=False,
        convert_to_int16=False,
        provide_offset_dists=True,
    )

    for parameters in list_parameters:

        shot_id = parameters["shot_id"]
        line_id = parameters["line_id"]
        rec_index = parameters["rec_index"]

        image_path = image_path_template.format(
            site_name=site_name, shot_id=shot_id, line_id=line_id, rec_index=rec_index
        )

        shot_traces = shot_line_gather_dataset.shot_to_trace_map[shot_id]
        line_traces = shot_line_gather_dataset.line_to_trace_map[line_id]
        line_gather_traces = np.intersect1d(shot_traces, line_traces)

        gather_ids = np.unique(
            [
                shot_line_gather_dataset.trace_to_gather_map[id]
                for id in line_gather_traces
            ]
        )
        assert len(gather_ids) == 1, "something is wrong"
        gather_id = gather_ids[0]

        datum = shot_line_gather_dataset[gather_id]

        rec_ids = datum["rec_ids"]
        rec_id = datum["rec_ids"][rec_index]

        m1 = evaluation_df["ShotId"] == shot_id
        m2 = evaluation_df["ReceiverId"].isin(rec_ids)
        gather_error_series = evaluation_df[m1 & m2].set_index("ReceiverId")["Errors"]

        datum["sampling_time_in_milliseconds"] = (
            shot_line_gather_dataset.samp_rate / 1000.0
        )

        list_prediction_time_in_milliseconds = []
        for ground_truth_label, rec_id in zip(datum["first_break_labels"], rec_ids):
            if rec_id in gather_error_series.index:
                error = gather_error_series[rec_id]
            else:
                error = np.NaN
            predicted_label = ground_truth_label + error
            prediction_timestamp = (
                predicted_label * shot_line_gather_dataset.samp_rate / 1000.0
            )
            list_prediction_time_in_milliseconds.append(prediction_timestamp)

        list_prediction_time_in_milliseconds = np.array(
            list_prediction_time_in_milliseconds
        )

        plotter = ReportSiteLineGatherWithTracePlotter(
            datum["sampling_time_in_milliseconds"], site_name
        )

        print(list_prediction_time_in_milliseconds[rec_index])

        fig, ax1, ax2 = plotter.create_figure(
            datum, list_prediction_time_in_milliseconds, rec_index
        )
        xmin = parameters["xmin"]
        xmax = parameters["xmax"]
        ax2.set_xlim(xmin, xmax)

        imin = int(xmin / datum["sampling_time_in_milliseconds"])
        imax = int(xmax / datum["sampling_time_in_milliseconds"])

        partial_trace = datum["samples"][rec_index, imin:imax]
        ax2.set_ylim(1.15 * np.min(partial_trace), 1.15 * np.max(partial_trace))
        fig.savefig(image_path)
        plt.close(fig)
