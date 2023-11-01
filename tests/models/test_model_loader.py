import os

import mock
import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    CheckpointConnector,
)
from torch.utils.data import Dataset, DataLoader

from hardpicks.models.model_loader import load_model
from hardpicks.models.fbp.unet import FBPUNet


@pytest.fixture()
def hyper_params():
    return dict(
        model_type="FBPUNet",
        model_checkpoint=None,
        unet_encoder_type="vanilla",
        unet_decoder_type="vanilla",
        coordconv=False,
        encoder_block_count=3,
        mid_block_channels=128,
        decoder_attention_type=None,
        decoder_block_channels=[64, 32, 16],
        use_full_metrics_during_training=True,
        update_scheduler_at_epochs=True,
        optimizer_type="Adam",
        scheduler_type="StepLR",
        max_epochs=2,
        batch_size=1,
        scheduler_params={"step_size": 10, "gamma": 0.1},
        eval_type="FBPEvaluator",
        eval_metrics=[],
        segm_first_break_prob_threshold=0.01,
        segm_class_count=2,
        use_first_break_prior=False,
        use_dist_offsets=False,
        use_checkpointing=False,
        gathers_to_display=0,
        loss_type="crossentropy",
        loss_params={},
    )


def create_dataloader(batch_size: int):
    class FakeDataset(Dataset):

        def __len__(self):
            return 128

        def __getitem__(self, item):
            torch.manual_seed(item)
            return {
                "samples": torch.rand(64, 64),
                "segmentation_mask": torch.randint(2, (64, 64))
            }

    return DataLoader(FakeDataset(), batch_size=batch_size, num_workers=0)


list_training_gpus = [0]  # will train on CPU and test on CPU by default
if torch.cuda.device_count() > 0:  # if GPU(s) are available, we'll train on those too
    list_training_gpus += [
        dev_count for dev_count in range(1, torch.cuda.device_count() + 1)
    ]


@pytest.fixture()
def trained_model_and_checkpoint_path(hyper_params, tmpdir, train_gpus):
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmpdir,
        filename="last-{epoch:03d}-{step:06d}",
        verbose=False,
    )
    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        torch.manual_seed(23423)
        initialized_model = FBPUNet(hyper_params)
        trainer = Trainer(
            max_epochs=hyper_params["max_epochs"],
            gpus=train_gpus,
            callbacks=[checkpoint_callback],
        )
        train_dataloader = create_dataloader(8 if train_gpus > 0 else 1)
        trainer.fit(initialized_model, train_dataloader=train_dataloader)
        checkpoint_connector = CheckpointConnector(trainer)
        checkpoint_connector.dump_checkpoint()
    checkpoint_filename = os.listdir(tmpdir)[0]
    checkpoint_path = os.path.join(tmpdir, checkpoint_filename)
    return initialized_model, checkpoint_path


def check_model_weights_are_the_same(model1, model2):
    model1_parameters = list(model1.parameters())
    model2_parameters = list(model2.parameters())

    assert len(model1_parameters) == len(model2_parameters)

    for lp, tp in zip(model1_parameters, model2_parameters):
        assert lp.data.shape == tp.data.shape
        assert np.allclose(lp.data, tp.data)


@pytest.mark.parametrize("train_gpus", list_training_gpus)
def test_load_model_with_checkpoint_path(
    trained_model_and_checkpoint_path, hyper_params
):
    trained_model, checkpoint_path = trained_model_and_checkpoint_path

    hyper_params_with_checkpoint = dict(hyper_params)
    hyper_params_with_checkpoint["model_checkpoint"] = checkpoint_path

    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        loaded_model = load_model(hyper_params_with_checkpoint)

    check_model_weights_are_the_same(trained_model, loaded_model)


@pytest.mark.parametrize("train_gpus", list_training_gpus)
def test_load_model_without_checkpoint_path(
    trained_model_and_checkpoint_path, hyper_params
):
    trained_model, _ = trained_model_and_checkpoint_path

    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        loaded_model = load_model(hyper_params)

    with pytest.raises(AssertionError):
        check_model_weights_are_the_same(trained_model, loaded_model)


@pytest.mark.parametrize("train_gpus", list_training_gpus)
def test_smoke_predict_from_loaded_model(
    trained_model_and_checkpoint_path, hyper_params, tmp_path
):
    trained_model, checkpoint_path = trained_model_and_checkpoint_path
    trainer = Trainer(max_epochs=3, gpus=0)

    hyper_params_with_checkpoint = dict(hyper_params)
    hyper_params_with_checkpoint["model_checkpoint"] = checkpoint_path

    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        loaded_model = load_model(hyper_params_with_checkpoint)

    dataloader = create_dataloader(1)
    # create a 'data_output_path' attribute, just like in the predict.py script, so that the
    # segm_base.on_predict_epoch_end method knows where to dump the test_dataframe.pkl (in this case, in the trash).
    setattr(dataloader, 'data_output_path', tmp_path.joinpath('test_dataframe.pkl'))

    predictions = trainer.predict(
        model=loaded_model,
        dataloaders=dataloader,
        return_predictions=True,
    )

    for prediction in predictions:
        assert not torch.any(prediction.isnan())
