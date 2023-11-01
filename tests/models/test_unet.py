import mock
import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch
import typing

import hardpicks.models.base as base
import hardpicks.models.losses as losses
import hardpicks.models.fbp.unet as unet
import hardpicks.models.unet_base as unet_base
import hardpicks.utils.hp_utils as hp_utils


def _get_vanilla_model_hyper_params(
    block_count: int,
    mid_block_channels: int = 128,
    segm_class_count: int = 2,
    loss_type: typing.AnyStr = "crossentropy",
):
    decoder_block_channels = [
        mid_block_channels // (2 ** block_idx) for block_idx in range(block_count)
    ]
    assert all([ch > 0 for ch in decoder_block_channels])
    return dict(
        unet_encoder_type="vanilla",
        unet_decoder_type="vanilla",
        encoder_block_count=block_count,
        mid_block_channels=mid_block_channels,
        decoder_block_channels=decoder_block_channels,
        decoder_attention_type=None,
        coordconv=False,
        segm_class_count=segm_class_count,
        segm_first_break_prob_threshold=0.5,
        use_dist_offsets=False,
        use_first_break_prior=False,
        gathers_to_display=0,
        max_epochs=5,
        optimizer_type="Adam",
        optimizer_params={},
        scheduler_type="StepLR",
        scheduler_params={},
        update_scheduler_at_epochs=True,
        loss_type=loss_type,
        loss_params={},
        use_checkpointing=False,
        use_full_metrics_during_training=True,
    )


def _get_model_without_hyperparam_checks(hparams_getter, **kwargs):
    # need to catch hyperparam logging, otherwise mlflow complains that they change...
    hyper_params = hparams_getter(**kwargs)
    with mock.patch.object(unet.FBPUNet, "save_hyperparameters"), \
            mock.patch.object(base.eval_loader, "get_evaluator"), \
            mock.patch.object(hp_utils, "log_hp"):
        model = unet.FBPUNet(hyper_params)
    return model


@pytest.mark.parametrize(
    "block_count", [1, 2, 4],
)
def test_unet_vanilla_block_count(block_count):
    hyper_params = _get_vanilla_model_hyper_params(block_count)
    # need to catch hyperparam logging, otherwise mlflow complains that they change...
    with mock.patch.object(unet.FBPUNet, "save_hyperparameters"), \
            mock.patch.object(base.eval_loader, "get_evaluator"), \
            mock.patch.object(hp_utils, "log_hp"):
        model = unet.FBPUNet(hyper_params)

    encoder = model.encoder
    assert isinstance(encoder, unet_base.Basic2DEncoder)
    assert isinstance(model.decoder, unet_base.Basic2DDecoder)
    assert encoder.block_count == block_count and len(encoder.blocks) == block_count
    assert len(model.decoder.blocks) == block_count
    assert model.decoder.mid_block_channels == 128
    assert model.decoder.head_class_count == 2


@pytest.mark.parametrize(
    "rows,cols", [(32, 32), (64, 64), (128, 32)],
)
def test_unet_vanilla_forward_without_warning(rows, cols):
    batch_size, orig_channels = 16, 1
    in_tensor = torch.randn(batch_size, orig_channels, rows, cols)
    model = _get_model_without_hyperparam_checks(_get_vanilla_model_hyper_params, block_count=2)
    out_tensor = model(in_tensor)
    assert not model.warned_bad_input_size_power2
    assert out_tensor.shape == (batch_size, 2, rows, cols)


@pytest.mark.parametrize(
    "rows,cols", [(40, 40), (160, 160), (224, 224)],
)
def test_unet_vanilla_forward_with_warning(rows, cols):
    batch_size, orig_channels = 16, 1
    in_tensor = torch.randn(batch_size, orig_channels, rows, cols)
    model = _get_model_without_hyperparam_checks(_get_vanilla_model_hyper_params, block_count=2)
    with mock.patch.object(unet_base, "logger") as fake_logger:
        out_tensor = model(in_tensor)
        assert fake_logger.warning.call_count == 1
    assert model.warned_bad_input_size_power2
    assert out_tensor.shape == (batch_size, 2, rows, cols)


@pytest.mark.parametrize(
    "loss_type", [loss_type for loss_type in losses.SUPPORTED_LOSS_TYPES]
)
def test_unet_with_all_losses(loss_type):
    batch_size, orig_channels, rows, cols = 16, 1, 256, 128
    in_tensor = torch.randn(batch_size, orig_channels, rows, cols)
    model = _get_model_without_hyperparam_checks(_get_vanilla_model_hyper_params, block_count=3)
    out_tensor = model(in_tensor)
    gt_tensor = torch.randint(0, 2, (batch_size, rows, cols))
    loss = model.loss_fn(out_tensor, gt_tensor)
    assert isinstance(loss, torch.Tensor) and loss.numel() == 1
    assert not np.isnan(loss.item())


def _get_resnet50_model_hyper_params():
    return dict(
        unet_encoder_type="resnet50",
        unet_decoder_type="vanilla",
        encoder_block_count=5,
        mid_block_channels=0,
        decoder_block_channels=[512, 256, 256, 64, 64],
        decoder_attention_type=None,
        coordconv=False,
        segm_class_count=2,
        segm_first_break_prob_threshold=0.5,
        use_dist_offsets=True,  # will make the encoder's 1st layer a 4-ch input
        use_first_break_prior=False,
        gathers_to_display=0,
        max_epochs=5,
        optimizer_type="Adam",
        optimizer_params={},
        scheduler_type="StepLR",
        scheduler_params={},
        update_scheduler_at_epochs=True,
        loss_type="crossentropy",
        loss_params={},
        use_checkpointing=False,
        use_full_metrics_during_training=True,
    )


def test_resunet50_forward():
    batch_size, orig_channels, rows, cols = 16, 4, 256, 128
    in_tensor = torch.randn(batch_size, orig_channels, rows, cols)
    model = _get_model_without_hyperparam_checks(_get_resnet50_model_hyper_params)
    encoder = model.encoder
    assert isinstance(encoder, smp.encoders.resnet.ResNetEncoder)
    out_tensor = model(in_tensor)
    assert not model.warned_bad_input_size_power2
    assert out_tensor.shape == (batch_size, 2, rows, cols)


def _get_efficientunetb0_model_hyper_params():
    return dict(
        unet_encoder_type="efficientnet-b0",
        unet_decoder_type="vanilla",
        encoder_block_count=5,
        mid_block_channels=0,
        decoder_block_channels=[512, 256, 256, 64, 64],
        decoder_attention_type=None,
        coordconv=False,
        segm_class_count=2,
        segm_first_break_prob_threshold=0.5,
        use_dist_offsets=True,  # will make the encoder's 1st layer a 4-ch input
        use_first_break_prior=False,
        gathers_to_display=0,
        max_epochs=5,
        optimizer_type="Adam",
        optimizer_params={},
        scheduler_type="StepLR",
        scheduler_params={},
        update_scheduler_at_epochs=True,
        loss_type="crossentropy",
        loss_params={},
        use_checkpointing=False,
        use_full_metrics_during_training=True,
    )


def test_efficientunet50_forward():
    batch_size, orig_channels, rows, cols = 16, 4, 256, 128
    in_tensor = torch.randn(batch_size, orig_channels, rows, cols)
    model = _get_model_without_hyperparam_checks(_get_efficientunetb0_model_hyper_params)
    encoder = model.encoder
    assert isinstance(encoder, smp.encoders.efficientnet.EfficientNetEncoder)
    out_tensor = model(in_tensor)
    assert not model.warned_bad_input_size_power2
    assert out_tensor.shape == (batch_size, 2, rows, cols)
