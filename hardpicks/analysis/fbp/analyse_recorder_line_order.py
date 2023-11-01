import matplotlib.pylab as plt
from tqdm import tqdm

from hardpicks import ANALYSIS_RESULTS_DIR
from hardpicks.analysis.fbp.analysis_plots import (
    plot_normalized_displacements,
)
from hardpicks.analysis.fbp.analysis_utils import get_data_from_site_dict
from hardpicks.data.fbp.site_info import get_site_info_array

if __name__ == "__main__":

    site_info_array = get_site_info_array()
    for site_dict in tqdm(site_info_array, desc="SITE"):

        fbp_data, site_name, data_file_path = get_data_from_site_dict(site_dict)

        measurement_df = fbp_data.get_measurement_dataframe()

        fig = plot_normalized_displacements(measurement_df)

        output_path = ANALYSIS_RESULTS_DIR.joinpath(
            f"{site_name}_normalized_displacements_along_receiver_lines.png"
        )
        fig.savefig(output_path)
        plt.close(fig)
