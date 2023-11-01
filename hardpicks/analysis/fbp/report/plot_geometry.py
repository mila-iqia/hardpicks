import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from hardpicks.analysis.fbp.report.path_constants import (
    style_path,
    pickles_directory,
    output_directory,
)
from hardpicks.analysis.fbp.analysis_plots import (
    plot_source_and_recorders,
)
from hardpicks.analysis.fbp.first_break_picking_seismic_data import (
    FirstBreakPickingSeismicData,
    good_shot_peg_per_site,
)
from hardpicks.analysis.logging_utils import (
    setup_analysis_logger,
)
from hardpicks.data.fbp.site_info import get_site_info_array

logger = logging.getLogger(__name__)
setup_analysis_logger()

plt.style.use(style_path)

# plots can be fiddly. It's faster to have the needed data on hand.
generate_pickles = False

pickle_path_template = os.path.join(
    str(pickles_directory), "{site_name}_{kind}_geometry.pkl"
)
image_path_template = os.path.join(str(output_directory), "{site_name}_geometry.png")

main_sites = [
    "Sudbury",
    "Halfmile",
    # "Matagami",
    "Brunswick",
    "Lalor",
]

if __name__ == "__main__":

    if generate_pickles:
        logger.info("Generating the stats pickles...")

        list_rows = []
        site_info_array = get_site_info_array()
        for site_info in tqdm(site_info_array, desc="SITE"):
            site_name = site_info["site_name"]
            if site_name not in main_sites:
                continue
            shot_peg_key = good_shot_peg_per_site[site_name]
            fbp_data = FirstBreakPickingSeismicData(
                site_info["processed_hdf5_path"],
                receiver_id_digit_count=site_info["receiver_id_digit_count"],
                shot_peg_key=shot_peg_key,
            )

            recorder_df = fbp_data.get_recorder_dataframe()
            source_df = fbp_data.get_source_dataframe()

            recorder_df.to_pickle(
                pickle_path_template.format(site_name=site_name, kind="recorder")
            )
            source_df.to_pickle(
                pickle_path_template.format(site_name=site_name, kind="source")
            )

    for site_name in main_sites:
        recorder_df = pd.read_pickle(
            pickle_path_template.format(site_name=site_name, kind="recorder")
        )
        source_df = pd.read_pickle(
            pickle_path_template.format(site_name=site_name, kind="source")
        )

        # if needed, apply hard-coded coordinates scaling based on value given in trace headers
        if site_name in ["Sudbury", "Brunswick"]:
            for coords in [recorder_df.x, recorder_df.y, source_df.x, source_df.y]:
                coords /= 100 if site_name == "Sudbury" else 10

        fig = plot_source_and_recorders(source_df, recorder_df, in_relative_coords=True)
        fig.suptitle(f"{site_name} Site: Arrangement of Geophones and Sources")
        fig.savefig(image_path_template.format(site_name=site_name))
        plt.close(fig)

    # Deal with the Sudbury SE of line 3.
    site_name = "Sudbury"
    recorder_df = pd.read_pickle(
        pickle_path_template.format(site_name=site_name, kind="recorder")
    )
    source_df = pd.read_pickle(
        pickle_path_template.format(site_name=site_name, kind="source")
    )

    mask = source_df.index <= 41
    reject_source_df = source_df[mask]

    coords_to_rescale = [
        recorder_df.x,
        recorder_df.y,
        source_df.x,
        source_df.y,
        reject_source_df.x,
        reject_source_df.y,
    ]
    for coords in coords_to_rescale:
        coords /= 100

    fig = plt.figure(figsize=(7.2, 4.45), dpi=200)
    ax = fig.add_subplot(111)

    fig.suptitle(f"{site_name} Site: Focus on Shot Line 3")

    origin = recorder_df.x.min(), recorder_df.y.min()
    rec_x, rec_y = recorder_df.x - origin[0], recorder_df.y - origin[1]
    ax.scatter(rec_x, rec_y, color="green", alpha=0.10, label="geophones")
    src_x, src_y = source_df.x - origin[0], source_df.y - origin[1]
    ax.scatter(src_x, src_y, color="red", alpha=0.10, label="sources")
    rej_src_x, rej_src_y = (
        reject_source_df.x - origin[0],
        reject_source_df.y - origin[1],
    )
    ax.scatter(rej_src_x, rej_src_y, color="red", alpha=1.0, label="cone shots")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc=0)

    xmin = 460000 - origin[0]
    xmax = 461500 - origin[0]
    ax.set_xlim([xmin, xmax])

    ymin = 5148000 - origin[1]
    ymax = 5150000 - origin[1]
    ax.set_ylim([ymin, ymax])

    image_path = os.path.join(
        str(output_directory), "Sudbury_geometry_focus_line_3.png"
    )

    fig.savefig(image_path)
    plt.close(fig)
