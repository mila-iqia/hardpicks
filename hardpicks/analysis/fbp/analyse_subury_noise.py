"""Plot the bad shots identified has having the shot cone noise."""
import yaml
import numpy as np
import matplotlib.pyplot as plt

from hardpicks import ANALYSIS_RESULTS_DIR, FBP_BAD_GATHERS_DIR
from hardpicks.analysis.fbp.first_break_picking_seismic_data import FirstBreakPickingSeismicData
from hardpicks.data.fbp.site_info import get_site_info_by_name

yaml_path = FBP_BAD_GATHERS_DIR.joinpath("bad-gather-ids_Sudbury_PLSC_cone_shot_noise_June10.yaml")
site_name = "Sudbury"

if __name__ == "__main__":
    with open(yaml_path, 'r') as f:
        rejected_gathers = yaml.load(f, Loader=yaml.FullLoader)[site_name]

    list_shot_ids = []
    for d in rejected_gathers.values():
        list_shot_ids.append(d['ShotId'])

    list_shot_ids = np.unique(list_shot_ids)

    site_info = get_site_info_by_name(site_name=site_name)

    fbp_data = FirstBreakPickingSeismicData(site_info["processed_hdf5_path"],
                                            receiver_id_digit_count=site_info["receiver_id_digit_count"],
                                            shot_peg_key='SHOTID')

    source_df = fbp_data.get_source_dataframe()
    recorder_df = fbp_data.get_recorder_dataframe()

    reject_source_df = source_df.loc[list_shot_ids]

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fig.suptitle("Sudbury geometry with shot annotations")

    for ax in [ax1, ax2]:
        ax.scatter(recorder_df.x, recorder_df.y, color="green", alpha=0.25, label="recorders")
        ax.scatter(source_df.x, source_df.y, color="orange", alpha=0.25, label="sources")
        ax.scatter(reject_source_df.x, reject_source_df.y, color="red", alpha=1., label="bad shots")

        ax.set_xlabel("X (UTM)")
        ax.set_ylabel("Y (UTM)")
        ax.legend(loc=0)

    for id, x, y in zip(source_df.index, source_df.x, source_df.y):
        ax2.annotate(f'{id}', (x, y))

    xmin = 46000000
    xmax = 46150000
    ax2.set_xlim([xmin, xmax])

    ymin = 514800000
    ymax = 515000000
    ax2.set_ylim([ymin, ymax])

    ax1.set_title('Global View')
    ax2.set_title('Zoom on Line 3')

    plt.savefig(ANALYSIS_RESULTS_DIR.joinpath("sudbury_noisy_shot_analysis.png"))
