"""PyTorch-Lightning Module loader.

In the regular cookie cutter, this file contains the code to create data loaders
directly. With PyTorch Lightning, the data loaders must be instantiated by a
Data Module object that also handles the split. This offers a proper way
to encapsulate the splitting procedure and to handle cross-validation. In this
case, the actual data module implementations are located in the subpackages for
each task (e.g. first break picking).

See https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.datamodule.html
for more information and examples.
"""

import logging
import typing

import pytorch_lightning

from hardpicks.data.fbp.data_module import FBPDataModule
import hardpicks.utils.hp_utils

logger = logging.getLogger(__name__)

SUPPORTED_MODULE_PAIRS = [("FBPDataModule", FBPDataModule)]


def _get_matching_module_class(
    module_name: typing.AnyStr,
) -> typing.Optional[typing.Type]:
    """Returns the type that matches a supported module, or ``None`` if no match is found."""
    for potential_module_name, module_class in SUPPORTED_MODULE_PAIRS:
        if potential_module_name == module_name:
            return module_class
    return None


def create_data_module(
    data_dir: typing.AnyStr,
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
) -> pytorch_lightning.LightningDataModule:
    """Instantiate a PyTorch-Lightning Data Module by name, given a parameter dict.

    Args:
        data_dir: the root data folder path that will be passed to the module constructor.
        hyper_params: hyperparameters from the config file.

    Returns:
        The PyTorch lightning data module that can create the actual data loaders.
    """
    hardpicks.utils.hp_utils.check_and_log_hp(
        names=["module_type"],
        hps=hyper_params,
    )
    module_type = hyper_params["module_type"]
    module_class = _get_matching_module_class(module_type)
    assert module_class is not None, f"unsupported module type: {module_type}"
    module = module_class(data_dir, hyper_params)
    hardpicks.utils.hp_utils.log_data_module_info(module)
    logger.info('module info:\n' + str(module) + '\n')
    return module
