import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

from hardpicks import PLOT_STYLE_PATH
from hardpicks.analysis.fbp.analysis_parser import (
    get_fbp_data,
    get_site_parser,
)
from hardpicks.analysis.fbp.publication.path_constants import (
    output_directory,
    pickles_directory,
)
from hardpicks.analysis.logging_utils import (
    configure_logger_for_console_only,
)
from hardpicks.plotting import PLEASANT_FIG_SIZE

logger = logging.getLogger(__name__)
configure_logger_for_console_only(logger)

plt.style.use(PLOT_STYLE_PATH)

pickle_path = pickles_directory.joinpath("normalized_trace_fourier_power.pkl")

# plots can be fiddly. It's faster to have the needed data on hand.
generate_pickles = False

image_path = output_directory.joinpath("fourier_power.png")

main_sites = [
    "Lalor",
    "Brunswick",
    "Halfmile",
    "Sudbury",
    # "Matagami",
]

color_map = dict(
    Brunswick="blue", Halfmile="red", Sudbury="green", Lalor="black", Matagami="grey"
)

data_base_dir = Path("/home/ubuntu/fbp_data/data/")

if __name__ == "__main__":

    if generate_pickles:
        data_dict = dict()
        for site_name in tqdm(main_sites, "SITE"):
            logger.info("Generating fbp data...")
            fbp_data = get_fbp_data(site_name, data_base_dir=data_base_dir)
            sampling_time_in_seconds = 0.001 * fbp_data.sample_rate_milliseconds
            time_in_ms = fbp_data.time_in_milliseconds
            sampling_frequency_in_hz = 1.0 / sampling_time_in_seconds
            del fbp_data

            logger.info("Generating dataset parser...")
            parser = get_site_parser(site_name, data_base_dir=data_base_dir)

            number_of_traces = 0

            total_power_spectrum = np.zeros(len(time_in_ms) // 2 + 1)
            indices = range(len(parser))
            for gather_idx in tqdm(indices, "GATHERS"):
                gather = parser[gather_idx]
                # The samples are normalized using the same strategy as in the experiments,
                # namely max amplitude = 1. Since this is what the models see, it makes sense
                # to compute the average Fourier of that, and not the raw data.
                gather_samples = gather["samples"]
                number_of_traces += len(gather_samples)

                freq, all_powers = ss.periodogram(
                    gather_samples, fs=sampling_frequency_in_hz, axis=-1
                )

                total_power_spectrum += all_powers.sum(axis=0)

            average_power_spectrum = total_power_spectrum / number_of_traces

            data_dict[site_name] = dict(
                freq=freq, average_power_spectrum=average_power_spectrum
            )

        with open(pickle_path, "wb") as f:
            pickle.dump(data_dict, f)

    with open(pickle_path, "rb") as f:
        data_dict = pickle.load(f)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    # Don't put a title on the figure, it will be described in the caption.
    # fig.suptitle("Normalized Power Spectrum for Random Subset of Traces")
    ax = fig.add_subplot(111)
    axins = inset_axes(
        ax,
        width=1.3,
        height=1.0,
        bbox_to_anchor=(-0.05, 0.05, 0.5, 0.5),
        bbox_transform=ax.transAxes,
        loc="center",
    )
    alpha = 0.75

    for site_name, site_data in data_dict.items():

        color = color_map[site_name]
        freq = site_data["freq"]
        average_power_spectrum = site_data["average_power_spectrum"]

        ax.semilogy(
            freq,
            average_power_spectrum,
            "-",
            lw=1,
            label=site_name,
            color=color,
            alpha=alpha,
        )

        axins.loglog(
            freq,
            average_power_spectrum,
            "-",
            lw=1,
            label=site_name,
            color=color,
            alpha=alpha,
        )

    ax.set_xlabel("Frequency (Hz)")
    axins.set_xlabel("Frequency (Hz)")

    ax.set_ylabel("Average Power Spectral Density (1/Hz)")
    ax.legend(loc=0)
    ax.set_xlim(xmin=0, xmax=500)
    axins.set_xlim(xmin=0, xmax=100)
    ax.set_ylim(ymin=1e-8, ymax=1e-2)
    axins.set_ylim(ymin=1e-6, ymax=1e-2)
    fig.tight_layout()

    fig.savefig(image_path)
    plt.close(fig)
