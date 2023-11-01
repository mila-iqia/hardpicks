from pathlib import Path

from hardpicks.data.fbp.site_info import get_site_info_array, site_info_keys


def test_all_sites_have_required_info(tmpdir):

    site_info_array = get_site_info_array(Path(tmpdir))
    required_site_info_keys = site_info_keys
    for site_info in site_info_array:
        for key in required_site_info_keys:
            assert key in site_info
