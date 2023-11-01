import logging
import os
import tempfile

import mock
import pytest

from hardpicks import FBP_DATA_DIR
from hardpicks.data.fbp.data_module import FBPDataModule
from hardpicks.data.fbp.site_info import get_site_info_by_name
from hardpicks.main import prepare_experiment
from tests.data.fbp import FBP_TEST_DIR


@pytest.mark.slow
class TestFBPDataModuleRealData:
    @pytest.fixture(scope="class")
    def output_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logging.info("Creating temporary directory for test")
            yield tmp_dir
        logging.info("Deleting temporary directory")

    @pytest.fixture(scope="class")
    def hyper_params(self, output_directory):
        config_file_path = str(FBP_TEST_DIR.joinpath("test_config.yaml"))
        run_name, exp_name, experiment_dir, hyper_params, config_file_backup_path = prepare_experiment(
            config_file_path, output_directory
        )
        return hyper_params

    @pytest.fixture(scope="class")
    def data_directory(self, hyper_params):
        # Validate that the real files exist and are in their default place
        for prefix in ["train", "valid", "test"]:
            list_loader_params = hyper_params[prefix + "_loader_params"]
            if list_loader_params is None:
                continue
            for loader_params in list_loader_params:
                site_name = loader_params["site_name"]
                site_info = get_site_info_by_name(site_name)
                if not os.path.exists(site_info["processed_hdf5_path"]):
                    return None
        return FBP_DATA_DIR

    @pytest.fixture()
    def data_module(self, hyper_params, data_directory, output_directory):
        if data_directory is None:
            pytest.skip("missing hdf5 file for real-data test. Skipping test.")

        # we need to prevent mlflow from logging hyperparameters
        patch_target = (
            "hardpicks.data.fbp.data_module.check_and_log_hp"
        )

        with mock.patch(patch_target), mock.patch.object(
            FBPDataModule, "_get_cache_dir", return_value=output_directory
        ):
            data_module = FBPDataModule(
                data_dir=data_directory, hyper_params=hyper_params
            )

            data_module.prepare_data()
            data_module.setup()
        return data_module

    def test_train_dataloader(self, data_module):
        data_loader = data_module.train_dataloader()
        # smoke test that we can get a sample from the dataloader
        for _ in data_loader:
            break
