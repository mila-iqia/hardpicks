import mock
import pytest
import torch

import hardpicks.models.resnet as new_resnet
import hardpicks.models.unet_base as unet_base
import torchvision.models.resnet as old_resnet


@pytest.fixture
def resnet18_config():
    return dict(
        block="BasicBlock",
        layers=[2, 2, 2, 2],
        channels=[64, 128, 256, 512],
        groups=1,
        width_per_group=64,
        norm_layer="BatchNorm",
    )


@pytest.fixture
def resnet34_config():
    return dict(
        block="BasicBlock",
        layers=[3, 4, 6, 3],
        channels=[64, 128, 256, 512],
        groups=1,
        width_per_group=64,
        norm_layer="BatchNorm",
    )


@pytest.fixture
def resnet50_config():
    return dict(
        block="Bottleneck",
        layers=[3, 4, 6, 3],
        channels=[64, 128, 256, 512],
        groups=1,
        width_per_group=64,
        norm_layer="BatchNorm",
    )


@pytest.fixture
def wideresnet50_config():
    return dict(
        block="Bottleneck",
        layers=[3, 4, 6, 3],
        channels=[64, 128, 256, 512],
        groups=1,
        width_per_group=128,
        norm_layer="BatchNorm",
    )


def _old_resnet_compatible_forward(x, model):
    # this returns the 'embedding' (pre-avgpool-and-fc tensor) of an old resnet impl
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return x


def _check_models_have_same_output(new_model, old_model):
    new_model.eval(), old_model.eval()
    new_model.load_state_dict(old_model.state_dict())
    input_tensor = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        new_embed = new_model(input_tensor)[-1]  # last featmap = 'embedding'
        old_embed = _old_resnet_compatible_forward(input_tensor, old_model)
    torch.testing.assert_allclose(new_embed, old_embed)


def test_resnet18(resnet18_config):
    old_resnet18 = old_resnet.resnet18(pretrained=False)
    new_resnet18 = new_resnet.ResNetEncoder(**resnet18_config)
    _check_models_have_same_output(new_resnet18, old_resnet18)


def test_resnet34(resnet34_config):
    old_resnet34 = old_resnet.resnet34(pretrained=False)
    new_resnet34 = new_resnet.ResNetEncoder(**resnet34_config)
    _check_models_have_same_output(new_resnet34, old_resnet34)


def test_resnet50(resnet50_config):
    old_resnet50 = old_resnet.resnet50(pretrained=False)
    new_resnet50 = new_resnet.ResNetEncoder(**resnet50_config)
    _check_models_have_same_output(new_resnet50, old_resnet50)


def test_wideresnet50(wideresnet50_config):
    old_wideresnet50 = old_resnet.wide_resnet50_2(pretrained=False)
    new_wideresnet50 = new_resnet.ResNetEncoder(**wideresnet50_config)
    _check_models_have_same_output(new_wideresnet50, old_wideresnet50)


@pytest.fixture
def narrowresnet34_config():
    return dict(
        block="BasicBlock",
        layers=[3, 4, 6, 3],
        channels=[32, 32, 64, 64],
        groups=1,
        width_per_group=64,
        norm_layer="BatchNorm",
    )


def test_narrowresnet34(narrowresnet34_config):
    narrowresnet34 = new_resnet.ResNetEncoder(**narrowresnet34_config)
    out_channels = narrowresnet34.out_channels  # identity + 5 stages
    exp_channels = narrowresnet34_config["channels"]
    assert out_channels == (3, exp_channels[0], *exp_channels)
    in_size = 64
    input_tensor = torch.randn(1, 3, in_size, in_size)
    with torch.no_grad():
        featmaps = narrowresnet34(input_tensor)
    assert len(featmaps) == len(out_channels)
    for downsample_step, (featmap, exp_ch_count) in enumerate(zip(featmaps, out_channels)):
        exp_out_size = in_size // (2 ** downsample_step)  # each stage downsamples by half
        assert featmap.shape == (1, exp_ch_count, exp_out_size, exp_out_size)


@pytest.fixture
def unet_config():
    return dict(
        unet_encoder_type="resnet",
        unet_decoder_type="vanilla",
        coordconv=False,  # must be false w/ current custom resnet impl
        use_checkpointing=False,
        encoder_block_channels=[64, 128, 128, 256],  # one less than real stage count, by definition
        mid_block_channels=0,  # must be 0 with a non-smp encoder
        decoder_block_channels=[256, 128, 128, 64, 64],
        decoder_attention_type=None,
        extra_encoder_params=dict(
            block="BasicBlock",
            layers=[2, 2, 2, 2],
        )
    )


def test_unet_with_custom_resnet_encoder(unet_config):
    expected_out_ch = 13
    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        encoder, decoder = unet_base.UNet._build_model(
            hyper_params=unet_config,
            encoder_input_channels=1,
            decoder_output_channels=expected_out_ch,
        )
    out_channels = encoder.out_channels  # identity + 5 stages
    exp_channels = unet_config["encoder_block_channels"]
    assert out_channels == (1, exp_channels[0], *exp_channels)
    in_size = 128
    input_tensor = torch.randn(1, 1, in_size, in_size)
    with torch.no_grad():
        featmaps = encoder(input_tensor)
        assert len(featmaps) == len(out_channels)
        assert all([f.shape[1] == ch for f, ch in zip(featmaps, out_channels)])
        out = decoder(featmaps)
    assert out.shape == (1, expected_out_ch, in_size, in_size)
