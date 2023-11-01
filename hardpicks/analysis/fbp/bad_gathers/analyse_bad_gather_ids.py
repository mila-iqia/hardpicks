"""This script reports the number of line gathers removed for each site."""
import yaml

from hardpicks import FBP_BAD_GATHERS_DIR

annotations_path = FBP_BAD_GATHERS_DIR.joinpath("bad-gather-ids_combined.yaml")

if __name__ == "__main__":
    with open(annotations_path, 'r') as f:
        rejected_gathers_dict = yaml.load(f, Loader=yaml.FullLoader)

    print("Site       number of rejected line gathers")
    for site_name in rejected_gathers_dict:
        number_of_rejects = len(rejected_gathers_dict[site_name])
        print(f"{site_name}   {number_of_rejects}")
