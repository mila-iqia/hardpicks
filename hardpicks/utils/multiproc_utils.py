import typing

import torch


def set_multiprocessing_start_method(
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
) -> None:
    """Sets the multiprocessing start method, if specified in the hyperparams dict."""
    start_method = hyper_params.get("multiproc_start_method", None)
    if start_method is not None:
        assert start_method in torch.multiprocessing.get_all_start_methods(), \
            f"unrecognized multiprocessing start method: {start_method}"
        if start_method != torch.multiprocessing.get_start_method(allow_none=None):
            torch.multiprocessing.set_start_method(start_method, force=True)
