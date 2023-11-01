from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec, patches

from hardpicks.analysis.fbp.first_break_picking_seismic_data import (
    FirstBreakPickingSeismicData,
)
from hardpicks.data.fbp.gather_preprocess import (
    ShotLineGatherPreprocessor,
)
from hardpicks.plotting import PLEASANT_FIG_SIZE


_GROUND_TRUTH_CONFIG_DICT = dict(
    s=20, facecolors="none", edgecolor="green", label="Ground Truth"
)
_ORIGINAL_GROUND_TRUTH_CONFIG_DICT = dict(
    s=10, facecolors="none", edgecolor="orange", label="Original Ground Truth"
)
_PREDICTIONS_CONFIG_DICT = dict(
    s=5, facecolors="red", edgecolor="none", label="Prediction"
)


def _get_bigger_symbol_size(input_config_dict: dict) -> dict:
    """A simple function to increase the size of symbols in config dict."""
    new_dict = dict(input_config_dict)
    if new_dict["edgecolor"] == "none":
        new_dict["s"] = 8 * input_config_dict["s"]
    else:
        new_dict["s"] = 4 * input_config_dict["s"]
    return new_dict


class GatherAndContextPlotter:
    """Gather and context plotter.

    It's tricky to produce nice visualizations for gathers, traces and contexts.
    This class wraps around common functionalities.
    """

    def __init__(
        self,
        fbp_data: FirstBreakPickingSeismicData,
        evaluator_df: Union[pd.DataFrame, None] = None,
    ):
        """Init.

        Args:
            fbp_data (FirstBreakPickingSeismicData): data objects that allows interacting with
                                                    the underlying hdf5 data store.

            evaluator_df (pd.DataFrame or None): results dataframe from the FBPEvaluator object. This
                                    is optional: if set to None, no prediction results will be available to plot.
        """
        self._fbp_data = fbp_data
        self.time_in_ms = fbp_data.time_in_milliseconds

        self._evaluator_df = evaluator_df

        self._show_original_ground_truth_picks = False

        self._show_ground_truth_picks = False
        self._remove_incorrect_ground_truth_picks = False

        self._show_predicted_picks = False

        self._original_first_break_pick_key = None

        # These flags control the 'multi-trace' contextual panes
        self._multitrace_context_around_ground_truth_picks = False
        self._context_range_in_ms_around_ground_truth_picks = None

        self._multitrace_context_around_predicted_picks = False
        self._context_range_in_ms_around_predicted_picks = None

        self._multitrace_context_width = None

        # These flags control the 'zoom-in' on trace of interest panes
        self._zoom_in_around_ground_truth_picks = False
        self._zoom_in_range_in_ms_around_ground_truth_picks = None

        self._zoom_in_around_predicted_picks = False
        self._zoom_in_range_in_ms_around_predicted_picks = None

    def show_original_ground_truth_picks(
        self, flag: bool = True, original_first_break_pick_key: str = "SPARE1"
    ):
        """Method to set flags."""
        # This option is only meaningful for a site where the picks have been modified (i.e., Lalor).
        self._show_original_ground_truth_picks = flag
        self._original_first_break_pick_key = original_first_break_pick_key

    def show_ground_truth_picks(
        self,
        flag: bool = True,
        remove_incorrect_ground_truth_pick: bool = True,
        context_range_in_ms=(-50, 200),
        zoom_in_range_in_ms=(-10, 10),
    ):
        """Method to set flags."""
        self._show_ground_truth_picks = flag
        self._remove_incorrect_ground_truth_picks = remove_incorrect_ground_truth_pick

        self._context_range_in_ms_around_ground_truth_picks = context_range_in_ms
        self._zoom_in_range_in_ms_around_ground_truth_picks = zoom_in_range_in_ms

    def show_predicted_picks(
        self,
        flag: bool = True,
        context_range_in_ms=(-50, 200),
        zoom_in_range_in_ms=(-10, 10),
    ):
        """Method to set flags."""
        self._show_predicted_picks = flag
        self._context_range_in_ms_around_predicted_picks = context_range_in_ms
        self._zoom_in_range_in_ms_around_predicted_picks = zoom_in_range_in_ms

    def show_multitrace_context(
        self, context_width=5, around_ground_truth=True, around_predictions=True
    ):
        """Method to set flags."""
        self._multitrace_context_around_ground_truth_picks = around_ground_truth
        self._multitrace_context_around_predicted_picks = around_predictions

        assert context_width % 2 == 1, "context width should be odd"
        self._multitrace_context_width = context_width

    def show_central_trace_zoom_in(
        self, around_ground_truth=True, around_predictions=True
    ):
        """Method to set flags."""
        self._zoom_in_around_ground_truth_picks = around_ground_truth
        self._zoom_in_around_predicted_picks = around_predictions

    def _build_fig_and_axes_dictionary(self):
        fig_and_axes_dict = dict()
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig_and_axes_dict["fig"] = fig
        gs = gridspec.GridSpec(2, 12)
        gs.update(wspace=0.125, hspace=0.35)

        if (
            self._zoom_in_around_ground_truth_picks
            or self._zoom_in_around_predicted_picks
        ):
            ax_row_slice = slice(0, 1)  # There will be two rows of axes
        else:
            ax_row_slice = slice(0, 2)  # There will be a single row of axes

        # Top row of axes
        if (
            self._multitrace_context_around_ground_truth_picks
            and self._multitrace_context_around_predicted_picks
        ):
            fig_and_axes_dict["line_gather_ax"] = fig.add_subplot(gs[ax_row_slice, 0:4])
            fig_and_axes_dict["context_ground_truth_ax"] = fig.add_subplot(
                gs[ax_row_slice, 4:8]
            )
            fig_and_axes_dict["context_predictions_ax"] = fig.add_subplot(
                gs[ax_row_slice, 8:12]
            )
        elif self._multitrace_context_around_ground_truth_picks:
            fig_and_axes_dict["line_gather_ax"] = fig.add_subplot(gs[ax_row_slice, 0:6])
            fig_and_axes_dict["context_ground_truth_ax"] = fig.add_subplot(
                gs[ax_row_slice, 6:12]
            )
        elif self._multitrace_context_around_predicted_picks:
            fig_and_axes_dict["line_gather_ax"] = fig.add_subplot(gs[ax_row_slice, 0:6])
            fig_and_axes_dict["context_predictions_ax"] = fig.add_subplot(
                gs[ax_row_slice, 6:12]
            )
        else:
            fig_and_axes_dict["line_gather_ax"] = fig.add_subplot(
                gs[ax_row_slice, 0:12]
            )

        # bottom row of axes (if any)
        if (
            self._zoom_in_around_ground_truth_picks
            and self._zoom_in_around_predicted_picks
        ):
            fig_and_axes_dict["zoom_in_ground_truth_ax"] = fig.add_subplot(gs[1, 0:5])
            fig_and_axes_dict["zoom_in_prediction_ax"] = fig.add_subplot(gs[1, 7:12])
        elif self._zoom_in_around_ground_truth_picks:
            fig_and_axes_dict["zoom_in_ground_truth_ax"] = fig.add_subplot(gs[1, 0:12])
        elif self._zoom_in_around_predicted_picks:
            fig_and_axes_dict["zoom_in_prediction_ax"] = fig.add_subplot(gs[1, 0:12])

        return fig_and_axes_dict

    def _get_line_gather_image_extent(self, number_of_traces):

        list_trace_index = np.arange(number_of_traces)
        left_limit = list_trace_index[0]
        right_limit = list_trace_index[-1]
        bottom_limit = self._fbp_data.time_in_milliseconds[-1]
        top_limit = self._fbp_data.time_in_milliseconds[0]
        extent = [left_limit, right_limit, bottom_limit, top_limit]
        return extent

    def _get_gather_indices(self, shot_peg, line_number):
        gather_indices = self._fbp_data.get_gather_indices(shot_peg, line_number)
        if len(gather_indices) == 0:
            raise ValueError(
                f"There is no gather for shot_peg = {shot_peg}, "
                f"line_number = {line_number}. Review Input!"
            )
        return gather_indices

    def _get_normalized_line_gather(self, shot_peg, line_number):
        gather_indices = self._get_gather_indices(shot_peg, line_number)
        raw_line_gather = self._fbp_data.raw_traces[gather_indices, :]
        normalized_line_gather = (
            ShotLineGatherPreprocessor.normalize_sample_with_tracewise_abs_max_strategy(
                raw_line_gather
            )
        )
        return normalized_line_gather

    def _get_ground_truth_first_break_picks_in_milliseconds(
        self, shot_peg, line_number
    ):
        gather_indices = self._get_gather_indices(shot_peg, line_number)
        ground_truth_fbp_in_ms = self._fbp_data.first_breaks_in_milliseconds[
            gather_indices
        ]
        return ground_truth_fbp_in_ms

    def _get_original_ground_truth_first_break_picks_in_milliseconds(
        self, shot_peg, line_number
    ):
        gather_indices = self._get_gather_indices(shot_peg, line_number)
        original_fbp_dataset = self._fbp_data.get_one_dimensional_dataset(
            self._original_first_break_pick_key
        )
        original_fbp_in_ms = original_fbp_dataset[gather_indices]
        return original_fbp_in_ms

    def _get_gather_dataframe(self, shot_peg, line_number):
        if self._evaluator_df is None:
            raise ValueError(
                "In order to extract model predictions, "
                "the evaluator dataframe must be loaded using method 'load_evaluator_dataframe'."
            )

        gather_indices = self._fbp_data.get_gather_indices(shot_peg, line_number)

        recorder_ids = self._fbp_data.record_pegs[gather_indices]

        shot_mask = self._evaluator_df["ShotId"] == shot_peg
        recorders_mask = self._evaluator_df["ReceiverId"].isin(recorder_ids)

        gather_df = self._evaluator_df[shot_mask & recorders_mask]

        assert len(gather_df) == len(
            recorder_ids
        ), "The number of traces is wrong in evaluator derived gather"
        np.testing.assert_equal(
            gather_df["ReceiverId"].values,
            recorder_ids,
            err_msg="The receivers order is wrong",
        )
        assert len(gather_df) == len(
            recorder_ids
        ), "The number of traces is wrong in evaluator derived gather"
        assert (
            len(gather_df["GatherId"].unique()) == 1
        ), "The gather id is not unique: something is wrong."

        return gather_df

    def _get_predicted_first_break_picks_in_milliseconds(self, shot_peg, line_number):
        gather_df = self._get_gather_dataframe(shot_peg, line_number)
        predicted_fbp_in_pixels = gather_df["Predictions"].values.astype(int)
        predicted_fbp_in_ms = (
            predicted_fbp_in_pixels * self._fbp_data.sample_rate_milliseconds
        )
        return predicted_fbp_in_ms

    def _plot_line_gather_ax(
        self, ax, normalized_line_gather, pick_data_dictionary, central_trace_index=None
    ):

        number_of_traces = len(normalized_line_gather)
        extent = self._get_line_gather_image_extent(number_of_traces)
        ax.imshow(
            normalized_line_gather.T,
            interpolation="none",
            extent=extent,
            cmap="Greys",
            aspect="auto",
        )
        for list_fbp_ms, config_dict in pick_data_dictionary.values():
            list_trace_indices = np.arange(len(list_fbp_ms))
            ax.scatter(list_trace_indices, list_fbp_ms, **config_dict)

        ax.set_ylabel("Time (ms)")
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.set_xticks([])
        xmin = 0
        xmax = number_of_traces - 1
        ymin = self._fbp_data.time_in_milliseconds[-1]
        ymax = self._fbp_data.time_in_milliseconds[0]

        if central_trace_index is not None:
            # Add a context box around the relevant traces
            width = self._multitrace_context_width
            height = ymax - ymin
            top_left_corner = (central_trace_index - width // 2 - 0.5, ymin)
            rect = patches.Rectangle(
                top_left_corner,
                width,
                height,
                linewidth=1,
                edgecolor="k",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)

        ax.legend(loc=0)

    def _filter_ground_truth_picks(self, ground_truth_picks_in_ms: np.array):

        filtered_picks = np.copy(ground_truth_picks_in_ms)

        if self._remove_incorrect_ground_truth_picks:
            invalid_pick_mask = np.where(ground_truth_picks_in_ms == -1)[0]
            filtered_picks[invalid_pick_mask] = -1000  # a large negative value that will put picks outside of image.
        return filtered_picks

    def _prepare_pick_data_dictionary(self, shot_peg, line_number):
        pick_data = dict()
        if self._show_ground_truth_picks:
            gt_fbp_ms = self._get_ground_truth_first_break_picks_in_milliseconds(
                shot_peg, line_number
            )
            pick_data["ground truth"] = (self._filter_ground_truth_picks(gt_fbp_ms), _GROUND_TRUTH_CONFIG_DICT)

        if self._show_predicted_picks:
            pred_fbp_ms = self._get_predicted_first_break_picks_in_milliseconds(
                shot_peg, line_number
            )
            pick_data["prediction"] = (pred_fbp_ms, _PREDICTIONS_CONFIG_DICT)

        if self._show_original_ground_truth_picks:
            original_fbp_ms = (
                self._get_original_ground_truth_first_break_picks_in_milliseconds(
                    shot_peg, line_number
                )
            )
            pick_data["original"] = (
                self._filter_ground_truth_picks(original_fbp_ms),
                _ORIGINAL_GROUND_TRUTH_CONFIG_DICT,
            )

        return pick_data

    @staticmethod
    def _prepare_data(
        pick_data_dictionary,
        central_trace_index,
        fig_and_axes_dict,
        list_booleans,
        list_pick_names,
        list_ranges,
        list_ax_names,
    ):
        data = []
        for flag, pick_name, range, ax_name in zip(
            list_booleans, list_pick_names, list_ranges, list_ax_names
        ):
            if flag:
                assert (
                    central_trace_index is not None
                ), "central trace index cannot be None."
                list_fbp_ms = pick_data_dictionary[pick_name][0]
                central_fbp_ms = list_fbp_ms[central_trace_index]
                xmin = central_fbp_ms + range[0]
                xmax = central_fbp_ms + range[1]
                data.append((xmin, xmax, fig_and_axes_dict[ax_name]))
        return data

    def _prepare_context_data(
        self, pick_data_dictionary, central_trace_index, fig_and_axes_dict
    ):
        list_booleans = [
            self._multitrace_context_around_ground_truth_picks,
            self._multitrace_context_around_predicted_picks,
        ]
        list_pick_names = ["ground truth", "prediction"]
        list_ranges = [
            self._context_range_in_ms_around_ground_truth_picks,
            self._context_range_in_ms_around_predicted_picks,
        ]
        list_ax_names = ["context_ground_truth_ax", "context_predictions_ax"]
        context_data = self._prepare_data(
            pick_data_dictionary,
            central_trace_index,
            fig_and_axes_dict,
            list_booleans,
            list_pick_names,
            list_ranges,
            list_ax_names,
        )

        return context_data

    def _prepare_zoom_in_data(
        self, pick_data_dictionary, central_trace_index, fig_and_axes_dict
    ):
        list_booleans = [
            self._zoom_in_around_ground_truth_picks,
            self._zoom_in_around_predicted_picks,
        ]
        list_pick_names = ["ground truth", "prediction"]
        list_ranges = [
            self._zoom_in_range_in_ms_around_ground_truth_picks,
            self._zoom_in_range_in_ms_around_predicted_picks,
        ]
        list_ax_names = ["zoom_in_ground_truth_ax", "zoom_in_prediction_ax"]
        zoom_in_data = self._prepare_data(
            pick_data_dictionary,
            central_trace_index,
            fig_and_axes_dict,
            list_booleans,
            list_pick_names,
            list_ranges,
            list_ax_names,
        )
        return zoom_in_data

    def _plot_multitrace_context_on_ax(
        self, normalized_line_gather, pick_data, central_trace_index, ax
    ):

        offset = self._multitrace_context_width // 2

        number_of_traces = len(normalized_line_gather)

        idx_min = np.max([central_trace_index - offset, 0])
        idx_max = np.min([central_trace_index + offset + 1, number_of_traces])

        for counter, idx in enumerate(range(idx_min, idx_max)):
            at_trace_of_interest = idx == central_trace_index
            if at_trace_of_interest:
                alpha = 1.0
            else:
                alpha = 0.35

            vertical_offset = offset * counter
            amplitude = normalized_line_gather[idx]
            amplitude_with_offset = amplitude + vertical_offset
            ax.plot(
                self._fbp_data.time_in_milliseconds,
                amplitude_with_offset,
                color="blue",
                alpha=alpha,
            )

            for list_fbp_in_ms, config_dict in pick_data.values():
                fbp_in_ms = list_fbp_in_ms[idx]
                first_break_amplitude = np.interp(
                    fbp_in_ms,
                    self._fbp_data.time_in_milliseconds,
                    amplitude_with_offset,
                )
                ax.scatter(
                    [fbp_in_ms],
                    [first_break_amplitude],
                    **_get_bigger_symbol_size(config_dict),
                )

        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlabel("Time (ms)")

    def _plot_zoom_in_on_ax(
        self, normalized_line_gather, pick_data_dictionary, central_trace_index, ax
    ):
        amplitude = normalized_line_gather[central_trace_index]
        ax.plot(self._fbp_data.time_in_milliseconds, amplitude, color="blue", alpha=1.0)

        for list_fbp_in_ms, config_dict in pick_data_dictionary.values():
            fbp_in_ms = list_fbp_in_ms[central_trace_index]
            first_break_amplitude = np.interp(
                fbp_in_ms, self._fbp_data.time_in_milliseconds, amplitude
            )
            ax.scatter(
                [fbp_in_ms],
                [first_break_amplitude],
                **_get_bigger_symbol_size(config_dict),
            )
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (ms)")

    def _get_yrange_for_zoom_in(self, amplitude, xrange):
        idx0 = int(xrange[0] / self._fbp_data.sample_rate_milliseconds)
        idx1 = int(xrange[1] / self._fbp_data.sample_rate_milliseconds)
        idx_min = np.max([idx0, 0])
        idx_max = np.min([idx1, len(amplitude)])
        if len(amplitude[idx_min:idx_max]) == 0:
            ymin = amplitude.min()
            ymax = amplitude.max()
        else:
            ymin = amplitude[idx_min:idx_max].min()
            ymax = amplitude[idx_min:idx_max].max()

        dy = 0.05 * (ymax - ymin)
        return ymin - dy, ymax + dy

    def _create_plot_title(
        self,
        shot_peg: int,
        line_number: int,
        central_trace_index: Union[int, None] = None,
    ):

        title = f"Shot {shot_peg}, Line {line_number}"

        if central_trace_index is not None:
            title += f", Trace {central_trace_index}"
            gather_df = self._get_gather_dataframe(shot_peg, line_number)

            gather_indices = self._fbp_data.get_gather_indices(shot_peg, line_number)
            recorder_id = self._fbp_data.record_pegs[
                gather_indices[central_trace_index]
            ]
            recorder_mask = gather_df["ReceiverId"] == recorder_id
            assert (
                recorder_mask.sum() == 1
            ), "There is not one and only one recorder id match."

            if "Errors" in gather_df.columns:
                error = gather_df[recorder_mask]["Errors"].values[0]
                title += f", Error = {int(error)} Pixels"
            if "Probabilities" in gather_df.columns:
                prob = 100 * gather_df[recorder_mask]["Probabilities"].values[0]
                title += f", P = {prob:3.1f} %"

        return title

    def get_trace_index_and_line_number(self, shot_peg: int, recorder_id: int) -> Tuple[int, int]:
        """Get trace index and line number from shot peg and recorder id.

        This convenience method extracts the "trace index", i.e., the rank of a given trace in a line gather,
        and the line number, based on the corresponding shot peg and recorder id. This is an adaptor method
        because there are many ways of representing the same information.

        Args:
            shot_peg (int): shot id
            recorder_id (int): identifier for the trace's recorder.

        Returns:
            line_number (int): line id for the line gather to show
            trace_index (int): rank of the trace in the line gather.
        """
        assert recorder_id in self._fbp_data.record_pegs, "unknown recorder id! Review input."

        matching_line_numbers = np.unique(self._fbp_data.record_line_numbers[self._fbp_data.record_pegs == recorder_id])
        assert len(matching_line_numbers) == 1, "Not one and exactly one line number match!?"
        line_number = matching_line_numbers[0]

        gather_indices = self._get_gather_indices(shot_peg, line_number)
        gather_record_pegs = self._fbp_data.record_pegs[gather_indices]
        assert recorder_id in gather_record_pegs, "The recorder id is not in the chosen line gather. Review input!"
        trace_idx = np.where(gather_record_pegs == recorder_id)[0][0]

        return line_number, trace_idx

    def generate_plot(
        self,
        shot_peg: int,
        line_number: int,
        central_trace_index: Union[int, None] = None,
        show_title: bool = True,
    ):
        """Generate Plot.

        This is the main plotting method, which generates a figure based on all the flags that are on or off.

        Args:
            shot_peg (int): shot id
            line_number (int): line id for the line gather to show
            central_trace_index (int or None): if we want to focus on a given trace for further contextual
                                representation, the index of the trace can be provided. This index is just
                                a positional integer on the line, not the receiver id.
            show_title (bool) : create a title for the figure.

        Returns:
            fig_and_axes_dict: a dictionary containing the figure and all the axes, should further plot fiddling
                    be desired.
        """
        fig_and_axes_dict = self._build_fig_and_axes_dictionary()

        normalized_line_gather = self._get_normalized_line_gather(shot_peg, line_number)
        pick_data_dictionary = self._prepare_pick_data_dictionary(shot_peg, line_number)

        self._plot_line_gather_ax(
            fig_and_axes_dict["line_gather_ax"],
            normalized_line_gather,
            pick_data_dictionary,
            central_trace_index,
        )

        context_data = self._prepare_context_data(
            pick_data_dictionary, central_trace_index, fig_and_axes_dict
        )

        for xmin, xmax, ax in context_data:
            self._plot_multitrace_context_on_ax(
                normalized_line_gather, pick_data_dictionary, central_trace_index, ax
            )
            ax.set_xlim(xmin, xmax)

        zoom_in_data = self._prepare_zoom_in_data(
            pick_data_dictionary, central_trace_index, fig_and_axes_dict
        )
        for xmin, xmax, ax in zoom_in_data:
            self._plot_zoom_in_on_ax(
                normalized_line_gather, pick_data_dictionary, central_trace_index, ax
            )
            amplitude = normalized_line_gather[central_trace_index]
            ymin, ymax = self._get_yrange_for_zoom_in(amplitude, (xmin, xmax))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        if show_title:
            title = self._create_plot_title(shot_peg, line_number, central_trace_index)
            fig_and_axes_dict["fig"].suptitle(title)

        return fig_and_axes_dict

    def plot_gather(self, shot_peg: int, line_number: int, show_title: bool = True):
        """Plot Gather.

        A convenience method to just plot the line gather, with sane defaults.

        Args:
            shot_peg (int): shot id
            line_number (int): line id for the line gather to show
            show_title (bool) : create a title for the figure.

        Returns:
            fig: Just the figure
            output_filename: suggested name for writing to disk.
        """
        output_filename = f"line_gather_shot_{shot_peg}_line_{line_number}.png"

        self.show_original_ground_truth_picks(flag=False)
        self.show_ground_truth_picks(flag=True)
        self.show_predicted_picks(flag=True)
        self.show_multitrace_context(
            around_ground_truth=False, around_predictions=False
        )
        self.show_central_trace_zoom_in(
            around_ground_truth=False, around_predictions=False
        )
        fig_and_ax_dict = self.generate_plot(
            shot_peg, line_number, central_trace_index=None, show_title=show_title
        )
        return fig_and_ax_dict["fig"], output_filename

    def plot_gather_and_context(
        self,
        shot_peg: int,
        line_number: int,
        central_trace_index: int,
        show_title: bool = True,
    ):
        """Plot Gather and context.

        A convenience method to plot the line gather and full context, with sane defaults.

        Args:
            shot_peg (int): shot id
            line_number (int): line id for the line gather to show
            central_trace_index (int): the index of the trace selected for further contextual representation.
            show_title (bool) : create a title for the figure.

        Returns:
            fig: Just the figure
            output_filename: suggested name for writing to disk.
        """
        output_filename = f"line_gather_and_context_shot_{shot_peg}_line_{line_number}_trace_{central_trace_index}.png"

        self.show_original_ground_truth_picks(flag=False)
        self.show_ground_truth_picks(flag=True)
        self.show_predicted_picks(flag=True)
        self.show_multitrace_context(around_ground_truth=True, around_predictions=True)
        self.show_central_trace_zoom_in(
            around_ground_truth=True, around_predictions=True
        )
        fig_and_ax_dict = self.generate_plot(
            shot_peg, line_number, central_trace_index, show_title=show_title
        )
        return fig_and_ax_dict["fig"], output_filename

    def plot_gather_and_minimal_context(
        self,
        shot_peg: int,
        line_number: int,
        central_trace_index: int,
        around_ground_truth: bool = True,
        show_title: bool = True,
    ):
        """Plot Gather and minimal context.

        A convenience method to plot the line gather and minimal context, with sane defaults.

        Args:
            shot_peg (int): shot id
            line_number (int): line id for the line gather to show
            central_trace_index (int): the index of the trace selected for further contextual representation.
            around_ground_truth (bool) : should the context traces be centered on the ground truth pick?
                if False, they will be centered on the predictions.
            show_title (bool) : create a title for the figure.

        Returns:
            fig: Just the figure
            output_filename: suggested name for writing to disk.
        """
        output_filename = \
            f"line_gather_and_minimal_context_shot_{shot_peg}_line_{line_number}_trace_{central_trace_index}.png"

        self.show_original_ground_truth_picks(flag=False)
        self.show_ground_truth_picks(flag=True)
        self.show_predicted_picks(flag=True)
        self.show_multitrace_context(around_ground_truth=around_ground_truth,
                                     around_predictions=not around_ground_truth)
        self.show_central_trace_zoom_in(
            around_ground_truth=False, around_predictions=False
        )
        fig_and_ax_dict = self.generate_plot(
            shot_peg, line_number, central_trace_index, show_title=show_title
        )
        return fig_and_ax_dict["fig"], output_filename

    def generate_multi_line_gather_plot(
        self,
        list_shot_peg_and_line_numbers: List[Tuple]
    ):
        """Generate multiple line gather plots as a single image.

        Args:
            list_shot_peg_and_line_numbers (List(tuple)): list of 4  (shot_peg, line_number)
             tuples to extract the line gathers to plot.

        Returns:
            fig: The matplotlib figure with the four sub-plots.
        """
        assert len(list_shot_peg_and_line_numbers) == 4, "4 line gathers at a time only."
        self.show_ground_truth_picks(True, remove_incorrect_ground_truth_pick=True)
        self.show_predicted_picks(True)

        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        list_axes = [fig.add_subplot(221 + i) for i in range(4)]

        for (shot_peg, line_number), ax in zip(list_shot_peg_and_line_numbers, list_axes):
            self._esthetic_plot_gather_on_ax(ax, line_number, shot_peg)

        for ax in list_axes[1:]:
            ax.get_legend().remove()

        fig.tight_layout()

        return fig

    def _esthetic_plot_gather_on_ax(self, ax, line_number, shot_peg):
        for spine in ax.spines.values():
            spine.set_visible(True)
        normalized_line_gather = self._get_normalized_line_gather(shot_peg, line_number)
        pick_data_dictionary = dict()
        for k, v in self._prepare_pick_data_dictionary(shot_peg, line_number).items():
            pick_data_dictionary[k] = (v[0], _get_bigger_symbol_size(v[1]))
        self._plot_line_gather_ax(ax,
                                  normalized_line_gather,
                                  pick_data_dictionary,
                                  central_trace_index=None)
        ax.set_yticks([])
        ax.set_ylabel("")

    def generate_esthetic_gather_plot(self, shot_peg: int, line_number: int, show_legend: bool = False):
        """Generate a single line gather plot with the same esthetic appearance as generate_multi_line_gather_plot.

        Args:
            shot_peg: shot peg value
            line_number: line number value
            show_legend: should the graph contain a legend?

        Returns:
            fig: The matplotlib figure with the four sub-plots.
            output_file_name: a suggested name for the output file.
        """
        output_filename = f"esthetic_line_gather_shot_{shot_peg}_line_{line_number}.png"
        self.show_ground_truth_picks(True, remove_incorrect_ground_truth_pick=True)
        self.show_predicted_picks(True)

        fig = plt.figure(figsize=(PLEASANT_FIG_SIZE[0] / 2, PLEASANT_FIG_SIZE[1] / 2))
        ax = fig.add_subplot(111)
        self._esthetic_plot_gather_on_ax(ax, line_number, shot_peg)

        if not show_legend:
            ax.get_legend().remove()

        fig.tight_layout()

        return fig, output_filename
