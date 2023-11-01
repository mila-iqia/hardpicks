import logging
import typing

import segmentation_models_pytorch as smp
import torch

logger = logging.getLogger(__name__)

SUPPORTED_LOSS_TYPES = [
    "dice", "focal", "lovasz", "softcrossentropy", "crossentropy",
]


class DiceLoss(smp.losses.DiceLoss):
    """Wrapper for Dice loss.

    This class is a wrapper around the SME dice loss to fix the
    issue of providing binary predictions as 2-channel logits.
    The original implementation expects no channels to be present.
    """
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute dice loss."""
        if self.mode == smp.losses.BINARY_MODE:
            # In binary mode, we'll always pick the logits of the last channel
            y_pred = y_pred[:, -1, :, :]
        return super().forward(y_pred, y_true)


class DiceCrossEntropyComboLoss(torch.nn.Module):
    """Wrapper for a combination of Dice + Cross Entropy losses.

    The 'alpha' blending factor will be applied to the dice loss, and (1-alpha) to the
    cross-entropy loss. All extra constructor kwargs will be forwarded to their corresponding
    loss module constructors.
    """

    def __init__(
        self,
        loss_mode: typing.Optional[typing.AnyStr] = None,
        loss_params: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
        ignore_index: typing.Optional[int] = None,
        print_loss_vals: bool = False,
    ):
        """Initializes the underlying loss modules and stores the blending factor to use."""
        super().__init__()
        if loss_params is None:
            loss_params = dict()
        self.dice_loss_fn = DiceLoss(
            mode=loss_mode,
            ignore_index=ignore_index,
            **loss_params.get("dice_loss_params", dict()),
        )
        self.crossentropy_loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            **loss_params.get("crossentropy_loss_params", dict()),
        )
        self.alpha = loss_params.get("alpha", 0.5)
        self.print_loss_vals = print_loss_vals or loss_params.get("print_loss_vals", False)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes and returns the (alpha)dice + (1-alpha)cross-entropy loss."""
        dice_loss = self.alpha * self.dice_loss_fn(y_pred, y_true)
        crossentropy_loss = (1 - self.alpha) * self.crossentropy_loss_fn(y_pred, y_true)
        if self.print_loss_vals:  # for debug purposes only
            logger.info(f"Scaled dice loss = {dice_loss.item():.05f}")
            logger.info(f"Scaled cross-entropy loss = {crossentropy_loss.item():.05f}")
        return dice_loss + crossentropy_loss


def get_loss_function(
    loss_type: typing.AnyStr,
    loss_mode: typing.Optional[typing.AnyStr] = None,
    loss_params: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
    ignore_index: typing.Optional[int] = None,
) -> typing.Callable:
    """Builds and returns the PyTorch-compatible loss function used in the training loop."""
    if loss_params is None:
        loss_params = {}
    assert isinstance(loss_params, dict)
    if loss_type == "jaccard":
        assert loss_mode is not None
        # TODO : JaccardLoss does not take an "ignore_index" argument. It's unclear why:
        #  it looks like they just forgot to add it in... We could implement it ourselves.
        assert ignore_index is None
        return smp.losses.JaccardLoss(
            mode=loss_mode,
            **loss_params,
        )
    elif loss_type == "dice":
        assert loss_mode is not None
        return DiceLoss(
            mode=loss_mode,
            ignore_index=ignore_index,
            **loss_params,
        )
    elif loss_type == "dice+crossentropy":
        assert loss_mode is not None
        return DiceCrossEntropyComboLoss(
            loss_mode=loss_mode,
            ignore_index=ignore_index,
            **loss_params,
        )
    elif loss_type == "focal":
        assert loss_mode is not None
        return smp.losses.FocalLoss(
            mode=loss_mode,
            ignore_index=ignore_index,
            **loss_params,
        )
    elif loss_type == "lovasz":
        assert loss_mode is not None
        return smp.losses.LovaszLoss(
            mode=loss_mode,
            ignore_index=ignore_index,
            **loss_params,
        )
    elif loss_type == "softcrossentropy":
        return smp.losses.SoftCrossEntropyLoss(
            ignore_index=ignore_index,
            **loss_params,
        )
    elif loss_type == "crossentropy":
        return torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            **loss_params,
        )
    elif loss_type == "mse":
        assert ignore_index is None
        return torch.nn.MSELoss(
            **loss_params,
        )
    else:
        raise NotImplementedError
