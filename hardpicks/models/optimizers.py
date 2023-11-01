"""Utility module that implements the optimizer getter function.

Might be extended later to do funky stuff for multi-optimizer setups/models, or to import
project-specific optimizer implementations.
"""
import importlib
import typing

import torch


def get_optimizer(
    optimizer_type: typing.AnyStr,
    optimizer_params: typing.Dict[typing.AnyStr, typing.Any],
    model_params: typing.Iterable,
) -> torch.optim.Optimizer:
    """Returns an optimizer object that should be used for backprop during training."""
    if not optimizer_type:
        return None  # in some cases, we might have no optimizer (e.g. when doing only inference)
    expected_default_prefix = "torch.optim."
    if not optimizer_type.startswith(expected_default_prefix) and "." not in optimizer_type:
        optimizer_type = expected_default_prefix + optimizer_type
    module_name, optimizer_name = optimizer_type.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    optimizer_type = getattr(module, optimizer_name)
    optimizer = optimizer_type(
        params=model_params,
        **optimizer_params,
    )
    return optimizer
