"""basic histograms of normalized distances in the extra channels.

We use the shot-to-recorder distance and the recorder-to-recorder distances (previous and next)
as features which we add in extra channels. these distances are normalized by dividing by 3000 m
for the shot-to-recorder distances and 50 m for the rec-to-rec distances.

This script plots the distributions of normalized distances for each site. We build simple
histograms by iterating over every traces in the cleaned, preprocessed dataset.
"""
import logging

import matplotlib.pyplot as plt
from tqdm import tqdm

from hardpicks import ANALYSIS_RESULTS_DIR
from hardpicks.analysis.logging_utils import (
    setup_analysis_logger,
)
from hardpicks.plotting import PLOT_STYLE_PATH, PLEASANT_FIG_SIZE


from hardpicks.data.fbp.gather_cleaner import (
    ShotLineGatherCleaner,
)
from hardpicks.data.fbp.gather_parser import (
    create_shot_line_gather_dataset,
)
from hardpicks.data.fbp.gather_preprocess import (
    ShotLineGatherPreprocessor,
)
from hardpicks.data.fbp.site_info import get_site_info_by_name

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
setup_analysis_logger()

list_site_names = [
    "Lalor",
    "Brunswick",
    "Halfmile",
    "Sudbury",
    # "Matagami",
]

if __name__ == "__main__":

    for site_name in list_site_names:
        logger.info(f"Starting process for site {site_name}")
        site_info = get_site_info_by_name(site_name)

        logger.info("Creating raw dataset")
        raw_dataset = create_shot_line_gather_dataset(
            hdf5_path=site_info["processed_hdf5_path"],
            site_name=site_info["site_name"],
            receiver_id_digit_count=site_info["receiver_id_digit_count"],
            first_break_field_name=site_info["first_break_field_name"],
            provide_offset_dists=True,
        )

        logger.info("Creating clean dataset")
        clean_dataset = ShotLineGatherCleaner(
            dataset=raw_dataset,
            auto_invalidate_outlier_picks=False,
        )

        logger.info("Creating preprocessed parser")
        parser = ShotLineGatherPreprocessor(clean_dataset, normalize_offsets=True)

        list_normalized_shot_to_rec_distance = []
        list_normalized_rec_to_rec_distance = []

        logger.info("Process each item in preprocessed dataset")

        # Here just iterating over the parser itself does not work. The code crashes
        # when we try to get an item beyond the number of items in the parser. Is there
        # a bug in how the parser length is defined?
        number_of_items = len(parser)
        for idx in tqdm(range(number_of_items), desc="ITEMS"):
            item = parser[idx]
            offset_distances = item["offset_distances"]
            list_normalized_shot_to_rec_distance.extend(list(offset_distances[:, 0]))
            list_normalized_rec_to_rec_distance.extend(list(offset_distances[:, 1]))
            list_normalized_rec_to_rec_distance.extend(list(offset_distances[:, 2]))

        logger.info("Plot distribution")
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

        fig.suptitle(f"Distribution of Normalized Distances for {site_name}")

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.hist(list_normalized_shot_to_rec_distance)
        ax2.hist(list_normalized_rec_to_rec_distance)

        ax1.set_xlabel("Normalized Shot-to-Recorder Dist.")
        ax2.set_xlabel("Normalized Recorder-to-Recorder Dist.")
        for ax in [ax1, ax2]:
            ax.set_ylabel("Count")

        fig.tight_layout()
        output_path = (
            ANALYSIS_RESULTS_DIR
            / f"normalized_distances_distribution_{site_name}.png"
        )
        fig.savefig(output_path)
        plt.close(fig)
