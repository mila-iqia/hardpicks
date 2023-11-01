"""First break picking (FBP) data subpackage."""

import os
import typing
from pathlib import Path
from hardpicks import FBP_DATA_DIR, FBP_ARTIFACTS_DIR, FBP_CACHE_DIR


def _get_valid_fbp_site_root_directories(
    data_dir: typing.Optional[typing.Union[Path, str]] = None,
):
    if data_dir is not None:
        data_dir = Path(data_dir)
        assert os.path.isdir(data_dir), \
            f"invalid data directory: {os.path.abspath(data_dir)}"
        artifacts_dir = data_dir.parent.joinpath("artifacts/")
        cache_dir = data_dir.parent.joinpath("cache/")
    else:
        data_dir = Path(FBP_DATA_DIR)
        artifacts_dir = Path(FBP_ARTIFACTS_DIR)
        cache_dir = Path(FBP_CACHE_DIR)
    return data_dir, artifacts_dir, cache_dir
