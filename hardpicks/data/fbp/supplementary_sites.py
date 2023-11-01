import typing
from pathlib import Path

from hardpicks.data.fbp import _get_valid_fbp_site_root_directories


def get_supplementary_lalor_site_info_array(
    data_dir: typing.Optional[typing.Union[Path, str]] = None,
):
    """Create array of site info dictionaries for the extra data provided for the Lalor site.

    Args:
        data_dir: directory where the raw site data can be found. The associated
            cache and artifact directories will be inferred at the same level. If
            ``None`` is given, the repo-level data directory will be used.

    Returns:
       site_info_array: list of dictionaries containing all site info.
    """
    data_dir, artifacts_dir, cache_dir = _get_valid_fbp_site_root_directories(data_dir)

    file_path = str(data_dir.joinpath("Lalor_3D/Lalor_raw_z_1500ms_norp_geom_v3.hdf5"))
    raw_md5_checksum = "d3c4722ef791cc3ec5adab656b16b925"
    receiver_id_digit_count = 3

    parameters = {
        "SPARE1": "original_picks",
        "SPARE2": "fine_tuning_bidirectional_5ms",
        "SPARE3": "fine_tuning_bidirectional_8ms",
        "SPARE4": "fine_tuning_forward_10ms",
    }

    lalor_site_info_array = []

    for first_break_field_name, site_name_postfix in parameters.items():
        lalor_site_info = dict(
            site_name=f"Lalor_{site_name_postfix}",
            first_break_field_name=first_break_field_name,
            raw_hdf5_path=file_path,
            processed_hdf5_path=file_path,
            raw_md5_checksum=raw_md5_checksum,
            processed_md5_checksum=raw_md5_checksum,
            receiver_id_digit_count=receiver_id_digit_count,
        )
        lalor_site_info_array.append(lalor_site_info)

    return lalor_site_info_array


def get_supplementary_subsampled_lalor_site_info_array(
    data_dir: typing.Optional[typing.Union[Path, str]] = None,
):
    """Create array of site info dictionaries for the extra data provided for the Lalor site sampled to 2ms.

    Args:
        data_dir: directory where the raw site data can be found. The associated
            cache and artifact directories will be inferred at the same level. If
            ``None`` is given, the repo-level data directory will be used.

    Returns:
       site_info_array: list of dictionaries containing all site info.
    """
    data_dir, artifacts_dir, cache_dir = _get_valid_fbp_site_root_directories(data_dir)

    file_path = str(data_dir.joinpath("Lalor_3D/Lalor_raw_z_1500ms_norp_geom_2ms.hdf5"))
    raw_md5_checksum = "cb3dc8bb24c98814bceb174ff514ab1a"
    receiver_id_digit_count = 3

    parameters = {
        "SPARE1": "original_picks",
        "SPARE2": "fine_tuning_bidirectional_5ms",
        "SPARE3": "fine_tuning_bidirectional_8ms",
        "SPARE4": "fine_tuning_forward_10ms",
    }

    lalor_site_info_array = []

    for first_break_field_name, site_name_postfix in parameters.items():
        lalor_site_info = dict(
            site_name=f"Lalor_2ms_{site_name_postfix}",
            first_break_field_name=first_break_field_name,
            raw_hdf5_path=file_path,
            processed_hdf5_path=file_path,
            raw_md5_checksum=raw_md5_checksum,
            processed_md5_checksum=raw_md5_checksum,
            receiver_id_digit_count=receiver_id_digit_count,
        )
        lalor_site_info_array.append(lalor_site_info)

    return lalor_site_info_array


def get_supplementary_subsampled_matagami_site_info_array(
    data_dir: typing.Optional[typing.Union[Path, str]] = None,
):
    """Create array of site info dictionaries for the extra data provided for the Matagami site sampled to 2ms.

    Args:
        data_dir: directory where the raw site data can be found. The associated
            cache and artifact directories will be inferred at the same level. If
            ``None`` is given, the repo-level data directory will be used.

    Returns:
       site_info_array: list of dictionaries containing all site info.
    """
    data_dir, artifacts_dir, cache_dir = _get_valid_fbp_site_root_directories(data_dir)

    file_path = str(data_dir.joinpath("Matagami_3D/MatagamiWest3D_part_shots_2ms.hdf5"))
    raw_md5_checksum = "3e4e5ee8f0e9a8eed12e7a4f0e5f21c8"
    receiver_id_digit_count = 3

    matagami_site_info = dict(
        site_name="Matagami_2ms",
        first_break_field_name="SPARE1",
        raw_hdf5_path=file_path,
        processed_hdf5_path=file_path,
        raw_md5_checksum=raw_md5_checksum,
        processed_md5_checksum=raw_md5_checksum,
        receiver_id_digit_count=receiver_id_digit_count,
    )
    matagami_site_info_array = [matagami_site_info]

    return matagami_site_info_array


def get_all_supplementary_sites_info_array(
    data_dir: typing.Optional[typing.Union[Path, str]] = None,
):
    """Get all the supplementary sites."""
    list_info_array = (
        get_supplementary_lalor_site_info_array(data_dir)
        + get_supplementary_subsampled_lalor_site_info_array(data_dir)
        + get_supplementary_subsampled_matagami_site_info_array(data_dir)
    )
    return list_info_array
