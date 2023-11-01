"""This module contains the high-level info necessary to parse each trace dataset."""
import typing
from pathlib import Path

from hardpicks.data.fbp import _get_valid_fbp_site_root_directories
from hardpicks.data.fbp.supplementary_sites import get_all_supplementary_sites_info_array

site_info_keys = [
    "site_name",
    "raw_hdf5_path",
    "processed_hdf5_path",
    "receiver_id_digit_count",
    "first_break_field_name",
    "raw_md5_checksum",
    "processed_md5_checksum",
]


def get_site_info_array(
    base_data_dir: typing.Optional[typing.Union[Path, str]] = None,
):
    """Create array of site info dictionary.

    Args:
        base_data_dir: directory where the raw site data can be found. The associated
            cache and artifact directories will be inferred at the same level. If
            ``None`` is given, the repo-level data directory will be used.

    Returns:
       site_info_array: list of dictionaries containing all site info.
    """
    data_dir, artifacts_dir, cache_dir = _get_valid_fbp_site_root_directories(base_data_dir)

    brunswick_site_info = dict(
        site_name="Brunswick",
        first_break_field_name="SPARE1",
        raw_hdf5_path=str(data_dir.joinpath("Brunswick_3D/Brunswick_orig_1500ms_V2.hdf5")),
        processed_hdf5_path=str(
            data_dir.joinpath("Brunswick_3D/Brunswick_orig_1500ms_V2.hdf5")
        ),
        raw_md5_checksum="3ca7b8d1633ec7cecc07f0eff94dff69",
        processed_md5_checksum="3ca7b8d1633ec7cecc07f0eff94dff69",
        receiver_id_digit_count=3,
    )

    halfmile_site_info = dict(
        site_name="Halfmile",
        first_break_field_name="SPARE1",
        raw_hdf5_path=str(
            data_dir.joinpath("Halfmile_3D/Halfmile3D_add_geom_sorted.hdf5")
        ),
        processed_hdf5_path=str(artifacts_dir.joinpath("preprocessed_Halfmile3D.hdf5")),
        raw_md5_checksum="dc7b0d181b8b81109a7e2e0ad60fa391",
        processed_md5_checksum="20e3586913c3fcc1c889e65b65e0f080",
        receiver_id_digit_count=4,
    )

    lalor_site_info = dict(
        site_name="Lalor",
        first_break_field_name="SPARE2",
        raw_hdf5_path=str(
            data_dir.joinpath("Lalor_3D/Lalor_raw_z_1500ms_norp_geom_v3.hdf5")
        ),
        processed_hdf5_path=str(data_dir.joinpath("Lalor_3D/Lalor_raw_z_1500ms_norp_geom_v3.hdf5")),
        raw_md5_checksum="d3c4722ef791cc3ec5adab656b16b925",
        processed_md5_checksum="d3c4722ef791cc3ec5adab656b16b925",
        receiver_id_digit_count=3,
    )

    sudbury_site_info = dict(
        site_name="Sudbury",
        first_break_field_name="SPARE1",
        raw_hdf5_path=str(data_dir.joinpath("Sudbury_3D/Sudbury3D_all_shots_2s.hdf")),
        processed_hdf5_path=str(
            data_dir.joinpath("Sudbury_3D/Sudbury3D_all_shots_2s.hdf")
        ),
        raw_md5_checksum="41e130ec2460344fdb64a992b6521fa5",
        processed_md5_checksum="41e130ec2460344fdb64a992b6521fa5",
        receiver_id_digit_count=3,
    )

    # NOTE: unavailable / not publicly released as of October 2023
    # matagami_site_info = dict(
    #     site_name="Matagami",
    #     first_break_field_name="SPARE1",
    #     raw_hdf5_path=str(
    #         data_dir.joinpath("Matagami_3D/MatagamiWest3D_part_shots.hdf5")
    #     ),
    #     processed_hdf5_path=str(artifacts_dir.joinpath("preprocessed_Matagami3D.hdf5")),
    #     raw_md5_checksum="31071009ab1bd2f09caaf2c44a690cee",
    #     processed_md5_checksum="df5e42d6ee8a039ea0f43f6bc45fd80e",
    #     receiver_id_digit_count=3,
    # )

    site_info_array = [
        sudbury_site_info,
        halfmile_site_info,
        # matagami_site_info,
        brunswick_site_info,
        lalor_site_info,
    ]
    site_info_array += get_all_supplementary_sites_info_array(base_data_dir)

    return site_info_array


def get_site_info_map(site_info_array: typing.List[typing.Dict]):
    """Get site info dictionary.

    Args:
        site_info_array: list of site_info dictionaries

    Returns:
        site_info_map: dictionary with site names as keys and site_info dicts as values
    """
    return {site_info["site_name"]: site_info for site_info in site_info_array}


def get_site_info_by_name(
    site_name: typing.AnyStr,
    data_dir: typing.Optional[typing.Union[Path, str]] = None,
):
    """Get a site_info dictionary by the name of the site.

    Args:
        site_name: name of the site
        data_dir: directory where the raw site data can be found. The associated
            cache and artifact directories will be inferred at the same level. If
            ``None`` is given, the repo-level data directory will be used.

    Returns:
        site_info: dictionary containing site information.
    """
    site_info_array = get_site_info_array(data_dir)
    site_info_map = get_site_info_map(site_info_array)

    return site_info_map[site_name]
