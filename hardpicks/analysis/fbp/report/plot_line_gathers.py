import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from hardpicks.analysis.fbp.report.path_constants import (
    style_path,
    pickles_directory,
    output_directory,
)
from hardpicks.analysis.fbp.velocity_analysis.line_gather_plotter import (
    ReportSiteLineGatherPlotter,
)
from hardpicks.analysis.logging_utils import (
    setup_analysis_logger,
)
from hardpicks.data.fbp.gather_parser import (
    create_shot_line_gather_dataset,
)
from hardpicks.data.fbp.site_info import get_site_info_by_name

logger = logging.getLogger(__name__)
setup_analysis_logger()

plt.style.use(style_path)

# plots can be fiddly. It's faster to have the needed data on hand.
generate_pickles = True

pickle_path_template = os.path.join(
    str(pickles_directory), "{site_name}_{shot_id}_{line_id}_datum.pkl"
)

image_path_template = os.path.join(
    str(output_directory), "{site_name}_{shot_id}_{line_id}_line_gather.png"
)


list_line_gathers = [
    dict(site_name="Sudbury", shot_id=1116, line_id=20),  # annotations are very wrong
    dict(site_name="Sudbury", shot_id=26, line_id=16),  # cone shot
    dict(site_name="Halfmile", shot_id=20221264, line_id=1009),  # line kink
    # dict(site_name="Matagami", shot_id=31066, line_id=1),
]

if __name__ == "__main__":

    if generate_pickles:
        dict_dataset = dict()
        for parameters in tqdm(list_line_gathers, desc="LG"):
            site_name = parameters["site_name"]
            shot_id = parameters["shot_id"]
            line_id = parameters["line_id"]

            site_info = get_site_info_by_name(site_name)
            if site_name in dict_dataset:
                shot_line_gather_dataset = dict_dataset[site_name]
            else:
                shot_line_gather_dataset = create_shot_line_gather_dataset(
                    hdf5_path=site_info["processed_hdf5_path"],
                    site_name=site_name,
                    first_break_field_name="SPARE1",
                    receiver_id_digit_count=site_info["receiver_id_digit_count"],
                    convert_to_fp16=False,
                    convert_to_int16=False,
                    provide_offset_dists=True,
                )

                dict_dataset[site_name] = shot_line_gather_dataset

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
            datum["sampling_time_in_milliseconds"] = (
                shot_line_gather_dataset.samp_rate / 1000.0
            )

            pickle_path = pickle_path_template.format(
                site_name=site_name, shot_id=shot_id, line_id=line_id
            )
            with open(pickle_path, "wb") as f:
                pickle.dump(datum, f)

            image_path_template = os.path.join(
                str(output_directory), "{site_name}_{shot_id}_{line_id}_line_gather.png"
            )

    for parameters in tqdm(list_line_gathers, desc="PLOT"):
        site_name = parameters["site_name"]
        shot_id = parameters["shot_id"]
        line_id = parameters["line_id"]

        pickle_path = pickle_path_template.format(
            site_name=site_name, shot_id=shot_id, line_id=line_id
        )
        with open(pickle_path, "rb") as f:
            datum = pickle.load(f)

        plotter = ReportSiteLineGatherPlotter(
            datum["sampling_time_in_milliseconds"], site_name
        )

        fig, ax = plotter.create_figure(datum)
        image_path = image_path_template.format(
            site_name=site_name, shot_id=shot_id, line_id=line_id
        )
        fig.savefig(image_path)
        plt.close(fig)
