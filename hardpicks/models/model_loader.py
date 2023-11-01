"""PyTorch-Lightning Module loader.

In the regular cookie cutter, this file contains the code to load optimizers,
schedulers, loss functions, and models. With PyTorch Lightning, the optimizer,
scheduler, and loss become the responsibility of the lightning module itself,
so each model implementation can be coupled with what objects it really needs.
The code for those is thus moved into the module implementation directly.

See https://pytorch-lightning.readthedocs.io/en/0.7.3/lightning-module.html
for more information and examples.

Note that our LightningModules should *NOT* try to create the data loaders on
their own. Instead, we will rely on an external data module that will be passed
to the PyTorch-Lightning trainer object.
"""
import functools
import logging
import os
import typing

import pytorch_lightning
import segmentation_models_pytorch as smp
import torch

from hardpicks.models.fbp.unet import FBPUNet
from hardpicks.models.resnet import ResNetEncoder as CustomResNet
import hardpicks.utils.hp_utils as hp_utils

logger = logging.getLogger(__name__)

SUPPORTED_SUPERVISED_MODELS = {"FBPUNet": FBPUNet}
"""The map of supported all-in-one models (with encoder+decoder/head included)."""

SUPPORTED_ENCODERS = {
    **{
        n: functools.partial(smp.encoders.get_encoder, name=n)
        for n in smp.encoders.get_encoder_names()
    },
    # the following encoder names all refer to the same custom resnet impl
    "CustomResNet": CustomResNet,
    "custom-resnet": CustomResNet,
    "resnet": CustomResNet,
}
"""The list of supported encoders; for now, we only rely on the SMP encoders."""

SUPPORTED_ENCODER_WRAPPERS = { }
"""The list of supported model wrappers (i.e. training/prediction/decoding heads)."""


def load_model(
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
) -> typing.Union[pytorch_lightning.LightningModule, torch.nn.Module]:
    """Instantiates (or reloads) a model by name, given a parameter dictionary.

    If the specified `model_type` is a supervised model trainer, it will be instantiated directly.
    Otherwise, an exception will be thrown.

    NOTE: if you intend to load several pretrained model outside an experiment (e.g. just to get
    predictions), MLFlow might hurl insults at you, as this function will call a bunch of
    hyperparameter logging functions which MLFlow does not allow us to call repeatedly. If you
    want to avoid this issue, mock `hardpicks.utils.hp_utils.log_hp`.

    Args:
        hyper_params: dictionary of hyperparameters to forward to the model constructor.

    Expected hyperparameters:
        model_type (string): the name of the model to instantiate; see the keys inside the
            `SUPPORTED_SUPERVISED_MODELS` map for a list of possibilities.
        model_checkpoint (optional): the path to a checkpoint to reload the model weights from
            (once instantiated). If missing or set to `None`, no weights will be reloaded.

    Returns:
        model: a model object.
    """
    hp_utils.check_and_log_hp(["model_type"], hyper_params)
    model_type = hyper_params["model_type"]
    assert model_type in [*SUPPORTED_SUPERVISED_MODELS, *SUPPORTED_ENCODER_WRAPPERS], \
        f"unsupported model type: {model_type}"
    assert model_type in SUPPORTED_SUPERVISED_MODELS
    model_class = SUPPORTED_SUPERVISED_MODELS[model_type]
    if not hyper_params.get("model_checkpoint", None):
        model = model_class(hyper_params)
    else:
        hp_utils.log_hp(["model_checkpoint"], hyper_params)
        model_ckpt_path = hyper_params["model_checkpoint"]
        assert os.path.isfile(model_ckpt_path), f"invalid checkpoint path: {model_ckpt_path}"
        # load the model from the checkpoint path with 'cpu' as the map_location; it will be up
        # to the training/prediction script to upload it back to the proper device (if needed)
        if issubclass(model_class, pytorch_lightning.LightningModule):
            model = model_class.load_from_checkpoint(model_ckpt_path, map_location="cpu")
        elif issubclass(model_class, torch.nn.Module):
            model = model_class(hyper_params)
            model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
        else:
            raise NotImplementedError
    hp_utils.log_model_info(model)
    return model


def create_encoder(
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
) -> torch.nn.Module:
    """Instantiates a PyTorch-based encoder by name, given a parameter dictionary.

    The returned module should be compatible with the classic `torch.nn.Module` interface
    (i.e. it should define a `forward` function). Compared to the `load_model` function defined
    above, this function CANNOT instantiate PyTorch-Lightning or CRF models; 'encoders' are instead
    assumed to be building blocks that are used in other bigger models, such as a ResNet backbone
    inside a CPC or SimCLR model.

    Args:
        hyper_params: dictionary of hyperparameters to parse for the encoder's info.

    Expected hyperparameters:
        encoder_type (string): the name of the encoder to instantiate; see the keys inside the
            `SUPPORTED_ENCODERS` map for a list of possibilities.
        encoder_params (optional): a dictionary of extra parameters to pass onto the encoder
            creation function. If it does not exist, no extra parameters will be passed.

    Returns:
        model: a `torch.nn.Module`-derived model object.
    """
    hp_utils.check_and_log_hp(["encoder_type"], hyper_params)
    encoder_type = hyper_params["encoder_type"]
    assert encoder_type in SUPPORTED_ENCODERS, f"unsupported encoder type: {encoder_type}"
    encoder_constr = SUPPORTED_ENCODERS[encoder_type]
    if "encoder_params" in hyper_params:
        hp_utils.log_hp(["encoder_params"], hyper_params)
        encoder_params = hyper_params["encoder_params"]
        if encoder_params is None:
            encoder_params = {}
        assert isinstance(encoder_params, dict)
    else:
        encoder_params = {}
    encoder = encoder_constr(**encoder_params)
    return encoder
