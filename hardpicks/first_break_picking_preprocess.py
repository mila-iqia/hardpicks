"""Create static preprocessed hdf5 files for first break picking.

This script is meant to be executed. It creates preprocessed hdf5 files
for certain sites by applying some data corrections. It checks that
all md5 checksums are as expected.
"""
import argparse
import logging

from hardpicks.analysis.logging_utils import (
    setup_analysis_logger,
)
from hardpicks.data.fbp.receiver_location_corrections import (
    preprocess_halfmile_dataset,
)
from hardpicks.data.fbp.site_info import (
    get_site_info_array,
    get_site_info_by_name,
)
# from hardpicks.data.fbp.wrong_receivers_removal import (
#     preprocess_matagami_dataset,
# )
from hardpicks.utils.hash_utils import get_hash_from_path
# from hardpicks.data.fbp.trace_parser import BASE_EXPECTED_HDF5_FIELDS

logger = logging.getLogger(__name__)
setup_analysis_logger()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_base_dir",
        help="Path to the top directory where the raw data is located.",
        required=True,
    )
    args = parser.parse_args()

    site_info_array = get_site_info_array(args.data_base_dir)

    halfmile_site_info = get_site_info_by_name("Halfmile", args.data_base_dir)
    # matagami_site_info = get_site_info_by_name("Matagami", args.data_base_dir)
    lalor_site_info = get_site_info_by_name("Lalor", args.data_base_dir)

    for site_info in site_info_array:
        site_name = site_info["site_name"]
        logger.info(f"Preprocessing site {site_name}...")

        raw_file_hash = get_hash_from_path(site_info["raw_hdf5_path"])
        assert (
            raw_file_hash == site_info["raw_md5_checksum"]
        ), "Raw md5 checksums do not match!"
        logger.info("Raw md5 checksums match.")

        if site_name == "Halfmile":
            logger.info("Creating the preprocessed Halfmile hdf5 file...")
            preprocess_halfmile_dataset(
                halfmile_site_info["raw_hdf5_path"],
                halfmile_site_info["processed_hdf5_path"],
            )
        # elif site_name == "Matagami":
        #     logger.info("Creating the preprocessed Matagami hdf5 file...")
        #     # to keep the original preprocessed hash, we need SPARE1 in its original place...
        #     hdf5_fields = BASE_EXPECTED_HDF5_FIELDS.copy()
        #     hdf5_fields.insert(-1, matagami_site_info["first_break_field_name"])
        #     preprocess_matagami_dataset(
        #         matagami_site_info["raw_hdf5_path"],
        #         matagami_site_info["processed_hdf5_path"],
        #         list_hdf5_fields=hdf5_fields,
        #     )

        processed_file_hash = get_hash_from_path(site_info["processed_hdf5_path"])
        assert (
            processed_file_hash == site_info["processed_md5_checksum"]
        ), f"Processed md5 checksums for site {site_name} do not match!"
        logger.info(f"Processed md5 checksums for site {site_name} match.")
