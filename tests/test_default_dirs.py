import os

import hardpicks


def test_default_dirs_exist():
    # note: all these directories should exist out-of-the-box with the repo

    paths_to_be_tested = [

        hardpicks.ROOT_DIR,
        hardpicks.TOP_DIR,

        hardpicks.CONFIG_DIR,
        hardpicks.EXAMPLES_DIR,
        hardpicks.DATA_ROOT_DIR,
        hardpicks.ANALYSIS_RESULTS_DIR,

        hardpicks.FBP_ROOT_DATA_DIR,
        hardpicks.FBP_BAD_GATHERS_DIR,
        hardpicks.FBP_ARTIFACTS_DIR,
        hardpicks.FBP_CACHE_DIR,
        hardpicks.FBP_FOLDS_DIR,

    ]

    for path in paths_to_be_tested:
        assert os.path.isdir(path), f"invalid path: {path}"
