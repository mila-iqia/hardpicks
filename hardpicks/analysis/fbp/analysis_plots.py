from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hardpicks.analysis.fbp.analysis_utils import get_base_ax_index, \
    get_deduplicated_displacement_dataframe
from hardpicks.utils.processing_utils import normalize_one_dimensional_array


def plot_basic_stats(stats_df: pd.DataFrame, line_gather_type: str = "Valid") -> plt.figure:
    """Plot site statistics."""
    alpha = 0.75

    fig = plt.figure(figsize=(7.2, 4.45))
    fig.suptitle(f"Basic Site Statistics for {line_gather_type} Line Gathers")

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.set_xlabel(f"Percentage of Invalid Picks\nin {line_gather_type} Line Gather")
    ax1.set_ylabel("Count")
    bad_pick_percentage_series = 100. * stats_df["bad picks"] / stats_df["gather size"]
    label = r"mean: {mean:2.1f} %".format(mean=bad_pick_percentage_series.mean())
    bad_pick_percentage_series.hist(ax=ax1, label=label, color='blue', alpha=alpha)
    ax1.set_xlim(0, 100)

    label_pattern = r"mean: {mean:2.1f}"
    ax2.set_xlabel(f"Number of Dead Traces\nin {line_gather_type} Line Gather")
    ax2.set_ylabel("Count")
    label = label_pattern.format(mean=stats_df["dead traces"].mean())
    stats_df["dead traces"].hist(ax=ax2, label=label, color='blue', alpha=alpha)

    ax3.set_xlabel(f"Number of Traces \nin {line_gather_type} Line Gather")
    ax3.set_ylabel("Count")
    label = label_pattern.format(mean=stats_df["gather size"].mean())
    stats_df["gather size"].hist(ax=ax3, label=label, color='blue', alpha=alpha)

    for ax in [ax1, ax2, ax3]:
        ax.legend(loc=0)
        ax.grid(False)

    fig.tight_layout()
    return fig


def plot_source_and_recorders(
    source_df: pd.DataFrame,
    recorder_df: pd.DataFrame,
    in_relative_coords: bool = False,
) -> plt.figure:
    """Plot configuration of sources and receivers for site."""
    fig = plt.figure(figsize=(7.2, 4.45), dpi=200)
    ax1 = fig.add_subplot(111)
    alpha = 0.75
    if in_relative_coords:
        origin = recorder_df.x.min(), recorder_df.y.min()
        rec_x, rec_y = recorder_df.x - origin[0], recorder_df.y - origin[1]
        ax1.scatter(rec_x, rec_y, s=5, color="green", label="geophones", alpha=alpha)
        src_x, src_y = source_df.x - origin[0], source_df.y - origin[1]
        ax1.scatter(src_x, src_y, s=5, color="red", label="sources", alpha=alpha)
        ax1.set_xlabel("Easting (m)")
        ax1.set_ylabel("Northing (m)")
    else:
        ax1.scatter(recorder_df.x, recorder_df.y, s=5, color="green", label="geophones", alpha=alpha)
        ax1.scatter(source_df.x, source_df.y, s=5, color="red", label="sources", alpha=alpha)
        ax1.set_xlabel("Easting Universal Transverse Mercator (m)")
        ax1.set_ylabel("Northing Universal Transverse Mercator (m)")
    ax1.legend(loc=0)
    return fig


def plot_line_by_line(recorder_df: pd.DataFrame) -> List[plt.figure]:
    """Plot each receiver line individually."""
    line_groups = recorder_df.groupby(by="line_number")

    list_figures = []
    for line_number, group_df in line_groups:
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"Arrangement of recorders and sources: line {line_number}")
        ax = fig.add_subplot(111)
        ax.scatter(
            recorder_df.x,
            recorder_df.y,
            color="grey",
            alpha=0.25,
            label="all recorders",
        )
        ax.scatter(group_df.x, group_df.y, color="green", label=f"line {line_number}")
        ax.set_xlabel("X (UTM)")
        ax.set_ylabel("Y (UTM)")
        ax.legend(loc=0)
        list_figures.append(fig)

    return list_figures


def plot_normalized_displacements(measurement_df: pd.DataFrame) -> plt.figure:
    """Plot the normalized displacements along receiver lines."""
    displacement_df = get_deduplicated_displacement_dataframe(measurement_df)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Normalized displacements along receiver lines for all shots")

    line_numbers = displacement_df["line_number"].unique()

    number_of_lines = len(line_numbers)
    number_of_rows, number_of_plots_per_row = get_base_ax_index(number_of_lines)

    for i, line_number in enumerate(line_numbers, start=1):

        ax = fig.add_subplot(number_of_rows, number_of_plots_per_row, i)

        ax.set_title(f"line number {line_number}")
        ax.set_aspect("equal")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])

        clean_line_df = displacement_df[displacement_df["line_number"] == line_number]

        ax.plot(
            clean_line_df["dx"],
            clean_line_df["dy"],
            "o",
            label=f"line number {line_number}",
        )
    return fig


def get_shot_figure(fbp_data, shot_peg, line_number):
    """Create figure that shows source/receiver geometry and subset of traces."""
    recorder_df = fbp_data.get_recorder_dataframe()
    source_df = fbp_data.get_source_dataframe()

    gather_indices = fbp_data.get_gather_indices(shot_peg, line_number)

    number_of_traces = len(gather_indices)

    raw_shot_gather = fbp_data.raw_traces[gather_indices, :]

    normalized_shot_gather = np.array(
        [normalize_one_dimensional_array(raw_trace) for raw_trace in raw_shot_gather]
    )

    shot_record_pegs = fbp_data.record_pegs[gather_indices]
    shot_fbp = fbp_data.first_breaks_in_milliseconds[gather_indices]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Shot {shot_peg} and Line {line_number}")
    ax1 = fig.add_subplot(221)
    ax1.set_title("Geometry")

    shot_source_df = source_df.loc[shot_peg]
    ax1.scatter(shot_source_df.x, shot_source_df.y, color="red", label="source")
    ax1.scatter(
        recorder_df.x, recorder_df.y, color="grey", alpha=0.25, label="recorders"
    )

    shot_receiver_df = recorder_df.loc[shot_record_pegs]
    ax1.scatter(
        shot_receiver_df.x,
        shot_receiver_df.y,
        color="green",
        label=f"shot gather pegs on line {line_number}",
    )

    ax1.set_xlabel("X (UTM)")
    ax1.set_ylabel("Y (UTM)")
    ax1.legend(loc=0)

    ax2 = fig.add_subplot(222)
    ax2.set_title(f"Subset of normalized traces along receiver line {line_number}")
    offset = 20
    ytick_labels = []
    ytick_positions = []
    list_fbp_time = []
    list_fbp_amplitude = []

    for counter, (normalized_trace, peg, fbp) in enumerate(
        zip(normalized_shot_gather[::10], shot_record_pegs[::10], shot_fbp[::10])
    ):
        vertical_offset = offset * counter
        ytick_positions.append(vertical_offset)
        ytick_labels.append(f"peg {peg}")
        amplitude = normalized_trace + vertical_offset
        ax2.plot(fbp_data.time_in_milliseconds, amplitude, color="blue")
        first_break_amplitude = np.interp(fbp, fbp_data.time_in_milliseconds, amplitude)
        list_fbp_time.append(fbp)
        list_fbp_amplitude.append(first_break_amplitude)

    ax2.scatter(list_fbp_time, list_fbp_amplitude, color="r", label="First Break Pick")

    ax2.set_xlabel("time (ms)")
    ax2.set_yticks(ytick_positions)
    ax2.set_yticklabels(ytick_labels)
    ax2.legend(loc=0)
    ax2.set_xlim(fbp_data.time_in_milliseconds[0], fbp_data.time_in_milliseconds[400])

    ax3 = fig.add_subplot(212)

    list_trace_index = np.arange(number_of_traces)

    left_limit = list_trace_index[0]
    right_limit = list_trace_index[-1]
    bottom_limit = fbp_data.time_in_milliseconds[-1]
    top_limit = fbp_data.time_in_milliseconds[0]

    extent = [left_limit, right_limit, bottom_limit, top_limit]

    ax3.set_ylabel("time (ms)")
    ax3.set_xlabel("trace along line (count)")

    image = ax3.imshow(
        normalized_shot_gather.T,
        interpolation="none",
        extent=extent,
        cmap="Greys",
        aspect="auto",
    )
    _ = fig.colorbar(image, extend="both", spacing="proportional", ax=ax3)
    ax3.scatter(
        list_trace_index,
        shot_fbp,
        s=20,
        facecolors="none",
        edgecolor="red",
        label="source",
    )

    ax3.set_ylim(ymax=top_limit)  # impose limit again to hide negative index picks

    fig.tight_layout()

    return fig
