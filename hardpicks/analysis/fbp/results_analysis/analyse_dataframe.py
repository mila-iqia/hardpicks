from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hardpicks.analysis.fbp.velocity_analysis.line_gather_plotter import (
    AbstractSiteLineGatherPlotter,
    PickProperties,
)
from hardpicks.data.fbp.gather_parser import (
    create_shot_line_gather_dataset,
)
from hardpicks.data.fbp.site_info import get_site_info_by_name


class PredictionLineGatherPlotter(AbstractSiteLineGatherPlotter):
    """WIP."""

    def create_figure(self, data_dict):
        """WIP."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        fig.suptitle(
            f"{self._site_name}: Shot {data_dict['shot_id']}, Line {data_dict['rec_line_id']}"
        )

        self._plot_line_gather(data_dict, fig, ax)

        self._plot_picks(data_dict, ax)

        ax.legend(loc=3)

        fig.tight_layout()

        return fig, ax

    @staticmethod
    def _get_pick_properties(data_dict):
        bad_mask = data_dict["bad_first_breaks_mask"]
        good_mask = ~bad_mask

        good_picks = PickProperties(
            mask=good_mask, size=30, symbol="o", color="green", label="good picks"
        )

        predicted_picks = PickProperties(
            mask=good_mask, size=30, symbol="+", color="yellow", label="predicted picks"
        )

        bad_picks = PickProperties(
            mask=bad_mask, size=30, symbol="o", color="red", label="bad picks"
        )

        return [good_picks, bad_picks, predicted_picks]

    def _plot_picks(self, data_dict, ax):
        first_break_times_in_milliseconds = data_dict["first_break_timestamps"]
        predicted_first_break_times_in_milliseconds = data_dict[
            "predicted_first_break_timestamps"
        ]

        number_of_traces = len(first_break_times_in_milliseconds)
        list_trace_index = np.arange(number_of_traces)

        good_picks, bad_picks, predicted_picks = self._get_pick_properties(data_dict)

        list_pick_properties = [good_picks, bad_picks, predicted_picks]
        list_first_break_picks = [
            first_break_times_in_milliseconds,
            first_break_times_in_milliseconds,
            predicted_first_break_times_in_milliseconds,
        ]

        for pick_properties, fbp in zip(list_pick_properties, list_first_break_picks):
            ax.scatter(
                list_trace_index[pick_properties.mask],
                fbp[pick_properties.mask],
                s=pick_properties.size,
                marker=pick_properties.symbol,
                facecolors=pick_properties.color,
                edgecolor=pick_properties.color,
                label=pick_properties.label,
            )


def get_hitrate(dataframe: pd.DataFrame, buffer_size_px: int = 1):
    """WIP."""
    # here, we threshold the absolute error value with a buffer to get the hit counts
    tot_count = dataframe["Errors"].count()
    hit_slice = dataframe["Errors"].abs() < buffer_size_px
    hit_slice[dataframe["Errors"].isnull()] = np.nan
    hit_count = hit_slice.sum()
    if tot_count > 0:
        assert hit_count <= tot_count, "nan masking probably messed up above"
        return hit_count / tot_count
    else:
        return None


dataframe_path = "/Users/bruno/monitoring/orion/foldA/mlruns/6/" \
    "8396efa76e0b43e29dd1b1e9e7161410/artifacts/dataframes/valid_dataframe_9.pkl"


if __name__ == "__main__":

    df = pd.read_pickle(dataframe_path)

    site_name = "Halfmile"
    site_info = get_site_info_by_name(site_name)

    data_file_path = Path(site_info["processed_hdf5_path"])

    shot_line_gather_dataset = create_shot_line_gather_dataset(
        site_info["processed_hdf5_path"],
        site_info["receiver_id_digit_count"],
        convert_to_fp16=False,
        convert_to_int16=False,
        provide_offset_dists=True,
    )

    shot_to_traces = [
        shot_line_gather_dataset.shot_to_trace_map[shot_id]
        for shot_id in df["ShotId"].values
    ]
    receiver_to_traces = [
        shot_line_gather_dataset.rec_to_trace_map[rec_id]
        for rec_id in df["ReceiverId"].values
    ]

    traces = [
        np.intersect1d(s, r)[0] for s, r in zip(shot_to_traces, receiver_to_traces)
    ]

    gather_ids = [shot_line_gather_dataset.trace_to_gather_map[t] for t in traces]
    df["GatherId"] = gather_ids

    z = df.groupby("GatherId")["Errors"].apply(
        lambda e: np.sqrt(np.sum(e ** 2)) / len(e)
    )

    # gather_id = 2916
    gather_id = 584

    sampling_time_in_milliseconds = shot_line_gather_dataset.samp_rate / 1000.0
    line_gather_plotter = PredictionLineGatherPlotter(
        sampling_time_in_milliseconds, site_name
    )

    datum_df = df[df["GatherId"] == gather_id]
    datum = shot_line_gather_dataset[gather_id]
    rec_ids = datum["rec_ids"]
    fbp_indices = datum["first_break_labels"]
    errors_df = datum_df[["ReceiverId", "Errors"]].set_index("ReceiverId")
    errors = errors_df.loc[rec_ids].values.flatten()
    prediction_indices = fbp_indices + errors
    datum["predicted_first_break_timestamps"] = (
        prediction_indices * sampling_time_in_milliseconds
    )
    fig, ax = line_gather_plotter.create_figure(datum)
