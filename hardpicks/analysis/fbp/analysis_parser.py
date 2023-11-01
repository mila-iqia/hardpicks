import logging

from hardpicks import ROOT_DIR
from hardpicks.analysis.fbp.first_break_picking_seismic_data import (
    FirstBreakPickingSeismicData,
    good_shot_peg_per_site,
)
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

logger = logging.getLogger(__name__)

_rejected_gather_yaml_path = (
    ROOT_DIR.parent / "data/fbp/bad_gathers/bad-gather-ids_combined.yaml"
)


def get_fbp_data(site_name, data_base_dir=None):
    """Get the fbp_data object for analysis, with correct flags built-in."""
    site_info = get_site_info_by_name(site_name=site_name, data_dir=data_base_dir)
    fbp_data = FirstBreakPickingSeismicData(
        path_to_hdf5_file=site_info["processed_hdf5_path"],
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        first_break_pick_key=site_info["first_break_field_name"],
        shot_peg_key=good_shot_peg_per_site[site_name],
    )
    return fbp_data


def get_site_parser(site_name, data_base_dir=None):
    """Get the parser for analysis, with correct flags built-in."""
    site_info = get_site_info_by_name(site_name=site_name, data_dir=data_base_dir)
    logger.info("Creating raw dataset")
    raw_dataset = create_shot_line_gather_dataset(
        hdf5_path=site_info["processed_hdf5_path"],
        site_name=site_info["site_name"],
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        first_break_field_name=site_info["first_break_field_name"],
        provide_offset_dists=False,
    )

    logger.info("Creating clean dataset")
    clean_dataset = ShotLineGatherCleaner(
        dataset=raw_dataset, rejected_gather_yaml_path=_rejected_gather_yaml_path
    )

    logger.info("Creating preprocessed parser")
    parser = ShotLineGatherPreprocessor(
        clean_dataset, normalize_samples=True, normalize_offsets=False
    )
    return parser
