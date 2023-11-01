"""Entrypoint for prediction using pretraining deep neural networks using PyTorch-Lightning."""
import logging
import os
import sys
import typing

import mock
import pytorch_lightning
import pytorch_lightning.callbacks
import pytorch_lightning.loggers
import torch
import yaml

from hardpicks.data.data_loader import create_data_module
from hardpicks.main import setup_arg_parser
from hardpicks.models.model_loader import load_model
from hardpicks.utils.hp_utils import check_config_contains_sigopt_tokens
from hardpicks.utils.prediction_utils import get_data_dump_path

logger = logging.getLogger(__name__)


def predict(args: typing.Optional[typing.Any] = None):
    """Predict loop implementation; take in a trained model and dataloader and output a data dump."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = setup_arg_parser()
    args = parser.parse_args(args)
    local_data_dir = args.data
    config_file_path = args.config
    use_gpu_if_available = args.use_gpu

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    with open(config_file_path, "r") as stream:
        hyper_params = yaml.load(stream, Loader=yaml.FullLoader)

    assert not check_config_contains_sigopt_tokens(hyper_params), \
        "prediction script cannot use hyperparam configs with sigopt tokens in them!"

    data_module = create_data_module(local_data_dir, hyper_params)
    data_module.prepare_data()
    data_module.setup()

    assert "model_checkpoint" in hyper_params, \
        "prediction script should be using hyperparam configs with assigned model checkpoints!"
    model_file_path = hyper_params["model_checkpoint"]
    assert os.path.exists(model_file_path), f"invalid checkpoint path: {model_file_path}"

    model = load_model(hyper_params)

    visible_gpu_count = torch.cuda.device_count()
    logger.info(f"visible gpu count: {visible_gpu_count}")
    if visible_gpu_count == 0:
        logger.info("no available GPU, will run on CPU!")
        use_gpu_if_available = False

    trainer = pytorch_lightning.Trainer(
        logger=False,
        accelerator="gpu" if use_gpu_if_available else "cpu",
        gpus=visible_gpu_count if use_gpu_if_available else 0,
        auto_select_gpus=True if use_gpu_if_available else False,
        precision=hyper_params["precision"] if use_gpu_if_available else 32,
        log_every_n_steps=hyper_params["log_every_n_steps"],
        num_sanity_val_steps=hyper_params["num_sanity_val_steps"],
        benchmark=hyper_params["set_benchmark"],
        deterministic=hyper_params["set_deterministic"],
    )

    # time to "predict" on all the data to log the final results/metrics now; we'll set the
    # appropriate secret attribute on the model to make this happen (see the `predict`-related
    # functions in models/base.py for more information)
    list_phase_names = ['train', 'valid', 'test', 'predict']
    list_data_loader_names = ['train_dataloader', 'val_dataloader', 'test_dataloader', 'predict_dataloader']
    loader_tuples = []
    for phase_name, data_loader_name in zip(list_phase_names, list_data_loader_names):
        logger.info(f"Adding dataloader '{data_loader_name}' to prediction list.")
        try:
            data_loader = getattr(data_module, data_loader_name)()
        except NotImplementedError:
            logger.info(f"Dataloader '{data_loader_name}' is not available. Moving on.")
            data_loader = None
        loader_tuples.append((phase_name, data_loader))

    output_preds = {}
    for dataloader_type, dataloader in loader_tuples:
        if dataloader is not None:
            output_data_dump_path = get_data_dump_path(model_file_path, dataloader_type, output_dir)
            assert hasattr(model, "predict_eval_output_path"), "missing secret attrib?"
            model.predict_eval_output_path = output_data_dump_path
            output_preds[dataloader_type] = trainer.predict(model=model, dataloaders=dataloader)

    return output_preds


if __name__ == '__main__':
    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        # make sure we don't log anything to mlflow
        predict()
