import logging
import tempfile
import numpy as np

import pytest

from hardpicks.data.fbp.gather_cleaner import (
    ShotLineGatherCleaner,
)
from hardpicks.data.fbp.gather_parser import (
    create_shot_line_gather_dataset,
)
from hardpicks.data.fbp.line_gather_cleaning_utils import (
    get_fitted_distances,
)
from tests.data.fbp.test_data_module import fake_params1, _get_list_site_info


@pytest.fixture()
def site_info():
    seed = 123
    site_name = "fake_data"
    with tempfile.TemporaryDirectory() as tmp_dir:
        logging.info("Creating temporary directory for test")
        fake_site_info = _get_list_site_info(
            [seed], [site_name], [fake_params1], tmp_dir
        )[0]
        yield fake_site_info

    logging.info("Deleting temporary directory")


@pytest.fixture()
def dataset(site_info):
    dataset = create_shot_line_gather_dataset(
        hdf5_path=site_info["processed_hdf5_path"],
        site_name=site_info["site_name"],
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        first_break_field_name=site_info["first_break_field_name"],
        convert_to_fp16=True,
        provide_offset_dists=True,
    )
    return dataset


@pytest.fixture()
def clean_dataset(dataset, outlier_detection_threshold):
    clean_dataset = ShotLineGatherCleaner(
        dataset,
        auto_invalidate_outlier_picks=True,
        outlier_detection_strategy="global-linear-fit-error-quantile",
        outlier_detection_threshold=outlier_detection_threshold,
        auto_fill_missing_picks=False,
    )
    return clean_dataset


@pytest.mark.parametrize("outlier_detection_threshold", [0.25, 0.5, 0.95])
def test_global_linear_fit_error_quantile_strategy(
    dataset, clean_dataset, outlier_detection_threshold
):

    list_gather_ids = range(len(dataset))
    for gather_id in list_gather_ids:
        data = dataset[gather_id]

        bad_mask = data["bad_first_breaks_mask"]
        times = data["first_break_timestamps"][~bad_mask].astype(np.float64)
        distances = data["offset_distances"][~bad_mask, 0].astype(np.float64)
        computed_distances = get_fitted_distances(
            clean_dataset.global_fit_velocity,
            clean_dataset.global_fit_time_offset,
            times,
        )

        max_absolute_error = np.abs(computed_distances - distances).max()

        if gather_id in clean_dataset.valid_gather_ids:
            assert (
                max_absolute_error < clean_dataset.absolute_error_threshold
            ), "The error is larger than expected"
        else:
            assert (
                max_absolute_error >= clean_dataset.absolute_error_threshold
            ), "The error is smaller than expected"
