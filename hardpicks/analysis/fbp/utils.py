import glob
import os
import typing

import mock
import pandas as pd
import pytorch_lightning
import torch
import yaml

from hardpicks import FBP_BAD_GATHERS_DIR, FBP_DATA_DIR
from hardpicks.data.data_loader import create_data_module
from hardpicks.models.model_loader import load_model
from hardpicks.utils.prediction_utils import get_data_dump_path


def predict_on_test_set(fold_root_path: typing.AnyStr,
                        config_file_path: typing.AnyStr,
                        model_checkpoint_path: typing.AnyStr,
                        test_site_name: typing.AnyStr,
                        first_break_threshold: float = 0.01):
    """Predict loop implementation; take in a trained model and dataloader and output a data dump."""
    # prepare the directory where we'll save the prediction result dataframe
    output_dir_path = os.path.join(fold_root_path, "test_output")
    os.makedirs(output_dir_path, exist_ok=True)
    dataframe_output_path = get_data_dump_path(model_checkpoint_path, "test", output_dir_path)
    if os.path.isfile(dataframe_output_path):
        return dataframe_output_path  # quick exit, we already have the dataframe we need
    # if we're here, we need to generate the predictions, but make sure we log nothing to mlflow
    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        # load the best checkpoint's configuration file and edit its content for the final eval
        with open(config_file_path, "r") as stream:
            hyper_params = yaml.load(stream, Loader=yaml.FullLoader)
        hyper_params["model_checkpoint"] = model_checkpoint_path
        hyper_params["segm_first_break_prob_threshold"] = first_break_threshold
        hyper_params["train_loader_params"] = []
        hyper_params["valid_loader_params"] = []
        bad_gathers_file_path = FBP_BAD_GATHERS_DIR.joinpath("bad-gather-ids_combined.yaml")
        hyper_params["test_loader_params"] = [{
            "site_name": test_site_name,
            "rejected_gather_yaml_path": bad_gathers_file_path,
            "use_cache": True,
            "normalize_samples": True,
            "normalize_offsets": True,
            "auto_invalidate_outliers": False,
        }]
        # prepare the data loader for the test set
        data_module = create_data_module(FBP_DATA_DIR, hyper_params)
        data_module.prepare_data()
        data_module.setup()
        # instantiate the model (with its 'best' weights)
        trained_model = load_model(hyper_params)
        # create a trainer object (it will only be used to generate predictions)
        trainer = pytorch_lightning.Trainer(
            logger=False,
            gpus=min(torch.cuda.device_count(), 1),
            auto_select_gpus=False,
            precision=hyper_params["precision"] if torch.cuda.device_count() > 0 else 32,
            amp_level="O1" if torch.cuda.device_count() > 0 else "O2",
            accelerator=None,
            log_every_n_steps=hyper_params["log_every_n_steps"],
            num_sanity_val_steps=hyper_params["num_sanity_val_steps"],
            benchmark=hyper_params["set_benchmark"],
            deterministic=hyper_params["set_deterministic"],
        )
        test_data_loader = data_module.test_dataloader()
        assert test_data_loader is not None
        setattr(test_data_loader, "data_output_path", dataframe_output_path)
        trainer.predict(model=trained_model, dataloaders=test_data_loader)
    return dataframe_output_path


def get_fold_config_and_ckpt_paths(fold_root_path: typing.AnyStr):
    """Returns the best checkpoint configuration and checkpoint paths for a specific fold."""
    config_file_path = os.path.join(fold_root_path, "artifacts/configs/*.config.yaml")
    config_file_path = glob.glob(config_file_path)
    assert len(config_file_path) == 1
    config_file_path = config_file_path[0]
    model_checkpoint_path = os.path.join(fold_root_path, "artifacts/checkpoints/best-*.ckpt")
    model_checkpoint_path = glob.glob(model_checkpoint_path)
    assert len(model_checkpoint_path) == 1
    model_checkpoint_path = model_checkpoint_path[0]
    return config_file_path, model_checkpoint_path


def rename_site_rms_dataframe_columns(site_rms_dataframe) -> pd.DataFrame:
    """Returns a copy of the site rms noise info dataframe so that columns can be intersected."""
    return site_rms_dataframe.rename(columns={
        "shot_id": "ShotId",
        "rec_id": "ReceiverId",
    })
