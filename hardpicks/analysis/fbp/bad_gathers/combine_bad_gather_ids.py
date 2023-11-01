"""This script combines the bad flagged annotations into a single config file."""
import yaml

from hardpicks import FBP_BAD_GATHERS_DIR
from hardpicks.data.fbp.gather_cleaner import ShotLineGatherCleaner
from hardpicks.data.fbp.gather_parser import create_shot_line_gather_dataset
from hardpicks.data.fbp.site_info import get_site_info_by_name


global_annotations_path = FBP_BAD_GATHERS_DIR.joinpath("bad-gather-ids_GB_May_28.yaml")
sudbury_annotations_path = FBP_BAD_GATHERS_DIR.joinpath("bad-gather-ids_Sudbury_PLSC_June10.yaml")
output_annotations_path = FBP_BAD_GATHERS_DIR.joinpath("bad-gather-ids_combined.yaml")

sudbury = 'Sudbury'
# By inspection, all shots with id equal or below 41 seem to be on the South East end of line 3, which
# is where "cone shots" were used. We want to remove all the corresponding line gathers.
shot_line_3_cutoff_id = 41

if __name__ == "__main__":

    with open(global_annotations_path, 'r') as f:
        global_rejected_gathers_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(sudbury_annotations_path, 'r') as f:
        sudbury_specific_rejected_gathers_dict = yaml.load(f, Loader=yaml.FullLoader)

    site_info = get_site_info_by_name(sudbury)
    raw_dataset = create_shot_line_gather_dataset(
        site_info["processed_hdf5_path"],
        sudbury,
        site_info["receiver_id_digit_count"],
        site_info["first_break_field_name"],
    )

    clean_dataset = ShotLineGatherCleaner(raw_dataset)
    bad_sudbury_gathers_dict = dict()

    # Identify the line 3 bad shots for Sudbury
    for idx in range(len(clean_dataset)):
        datum = clean_dataset.get_meta_gather(idx)
        if datum['shot_id'] <= shot_line_3_cutoff_id:
            gather_id = int(datum['gather_id'])
            shot_id = int(datum['shot_id'])
            rec_id = int(datum['rec_ids'][0])
            bad_sudbury_gathers_dict[gather_id] = {'ReceiverId': rec_id, 'ShotId': shot_id}

    # combine all the sudbury bad annotations
    bad_sudbury_gathers_dict.update(sudbury_specific_rejected_gathers_dict[sudbury])
    bad_sudbury_gathers_dict.update(global_rejected_gathers_dict[sudbury])

    # update the dictionary of all bad annotations
    combined_rejected_gathers_dict = dict(global_rejected_gathers_dict)
    combined_rejected_gathers_dict[sudbury] = bad_sudbury_gathers_dict

    with open(output_annotations_path, 'w') as f:
        yaml.dump(combined_rejected_gathers_dict, f)
