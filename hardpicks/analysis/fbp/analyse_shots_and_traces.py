import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from hardpicks import ANALYSIS_RESULTS_DIR
from hardpicks.analysis.fbp.analysis_plots import get_shot_figure
from hardpicks.analysis.fbp.analysis_utils import get_data_from_site_dict
from hardpicks.data.fbp.site_info import get_site_info_array

if __name__ == "__main__":

    np.random.seed(0)

    site_info_array = get_site_info_array()

    for site_dict in tqdm(site_info_array, desc="SITE"):

        fbp_data, site_name, data_file_path = get_data_from_site_dict(site_dict)

        unique_shot_pegs = np.unique(fbp_data.shot_pegs)
        unique_lines_numbers = np.unique(fbp_data.record_line_numbers)

        list_shot_pegs = np.random.choice(unique_shot_pegs, 2)

        list_image_paths = []
        for shot_peg in tqdm(list_shot_pegs, desc="SHOT PEG"):
            for line_number in unique_lines_numbers:

                gather_indices = fbp_data.get_gather_indices(shot_peg, line_number)
                if len(gather_indices) == 0:
                    continue

                fig = get_shot_figure(fbp_data, shot_peg, line_number)

                filename = f"traces_shot_{shot_peg}_line_{line_number}.png"
                output_path = ANALYSIS_RESULTS_DIR.joinpath(filename)
                fig.savefig(output_path)
                plt.close(fig)
                list_image_paths.append(output_path)

        # pdf_output_path = ANALYSIS_RESULTS_DIR.joinpath(f"{site_name}_shot_gathers.pdf")
        # create_pdf_from_list_of_images(list_image_paths, pdf_output_path)
        for path in list_image_paths:
            path.unlink()
