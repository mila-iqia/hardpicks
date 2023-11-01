"""Script Description.

This script is meant to visualize the receivers and sources in 3D plots to see if there are
any obvious problems. It is nearly impossible to see anything without rotating these images
dynamically, so it does not make sense to produce flat articacts for a sanity check report: this
must be done interactively.
"""
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from hardpicks.analysis.fbp.analysis_utils import get_data_from_site_dict
from hardpicks.data.fbp.site_info import get_site_info_array


def create_3D_figure(df: pd.DataFrame):
    """Plot the points in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = df["x"].values
    ys = df["y"].values
    zs = df["z"].values
    ax.scatter(xs, ys, zs, marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")

    return fig


if __name__ == "__main__":

    site_info_array = get_site_info_array()
    for site_dict in site_info_array:

        fbp_data, site_name, data_file_path = get_data_from_site_dict(site_dict)

        recorder_df = fbp_data.get_recorder_dataframe()
        source_df = fbp_data.get_source_dataframe()

        for df, kind in zip([recorder_df, source_df], ["receiver", "source"]):
            fig = create_3D_figure(df)
            fig.suptitle(f"{kind} positions for site {site_name}")

    plt.show()
