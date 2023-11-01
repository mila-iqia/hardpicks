"""Utility module that implements the scheduler getter function.

Might be extended later to do funky stuff for multi-scheduler setups/models, or to import
project-specific scheduler implementations.
"""
import importlib
import math
import typing

import torch.optim
import torch.optim.lr_scheduler


def get_scheduler(
    scheduler_type: typing.AnyStr,
    scheduler_params: typing.Dict[typing.AnyStr, typing.Any],
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    step_count_per_epoch: int,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Returns a scheduler that should be used to adjust the learning rate during training."""
    assert isinstance(optimizer, torch.optim.Optimizer)
    if scheduler_type == "LinearWarmupCosineAnnealingLR":
        scheduler_type = LinearWarmupCosineAnnealingLR
        assert "warmup_ratio" in scheduler_params, "missing required argument: 'warmup_ratio'"
        warmup_ratio = scheduler_params["warmup_ratio"]
        assert 0 <= warmup_ratio <= 1
        max_step_count = max_epochs * step_count_per_epoch
        assert "warmup_steps" not in scheduler_params and "max_steps" not in scheduler_params
        scheduler_params = {
            "warmup_steps": int(warmup_ratio * max_step_count),
            "max_steps": max_step_count,
            **{key: val for key, val in scheduler_params.items() if key != "warmup_ratio"},
        }
    else:
        expected_default_prefix = "torch.optim.lr_scheduler."
        if not scheduler_type.startswith(expected_default_prefix) and "." not in scheduler_type:
            scheduler_type = expected_default_prefix + scheduler_type
        module_name, scheduler_name = scheduler_type.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_name)
        scheduler_type = getattr(module, scheduler_name)
    scheduler = scheduler_type(
        **scheduler_params,
        optimizer=optimizer,
    )
    return scheduler


# TODO: Consider replacing with pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR?
class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Returns a scheduler that provides a linear LR warmup followed by cosine annealing.

    NOTE1: this schedule should be called PER ITERATION in order to get accurate (and hopefully not
    all-zero learning rates) across all epochs.

    NOTE2: this implementation might not play nice with other schedulers in case you want to combine
    them (see the note in the `get_lr` implementation below).
    """

    must_step_per_iter = True  # for a couple of asserts in the training loop only

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,  # if we bust this, the LR will remain at the minimum (likely zero!)
        peak_ratio: float = 1.0,  # at the peak, multiply the LR by this value
        last_epoch: int = -1,
    ) -> None:
        """Stores the scheduling parameters and calls the base class constructor."""
        # NOTE: the beginning/end learning rates will ALWAYS be zero in this implementation
        assert warmup_steps <= max_steps
        assert 0 < peak_ratio
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.peak_ratio = peak_ratio
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> typing.List[float]:
        """Returns the learning rate in the format expected by PyTorch's base class."""
        # note1: many base class attribs are suffixed 'epoch' below, just pretend they're 'step'
        # note2: we don't use the `group["lr"]` adjustment approach here, so stacking schedulers
        #        will probably break (i.e. we derive the LRs from the base rate, not the current)
        if self.last_epoch < 0:  # before '0th' step, LR = zero (constant)
            return [0.0] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_steps:  # during warmup, we slowly scale up to the peak
            return [
                base_lr * self.peak_ratio * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch == self.warmup_steps:  # this is the peak
            return [base_lr * self.peak_ratio for base_lr in self.base_lrs]
        elif self.last_epoch < self.max_steps:
            return [  # after warmup, we anneal down to the final LR
                self._get_annealing_val(
                    self.last_epoch - self.warmup_steps,
                    self.max_steps - self.warmup_steps,
                ) * base_lr * self.peak_ratio
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch >= self.max_steps:  # if we're past the end, LR = zero (constant)
            return [0.0] * len(self.base_lrs)
        raise NotImplementedError

    @staticmethod
    def _get_annealing_val(step: int, nmax: int):
        """Internal utility function for the cosine annealing factor."""
        return (1 + math.cos(math.pi * step / nmax)) / 2

    def _get_closed_form_lr(self) -> typing.List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        raise NotImplementedError  # this should never be called anymore, it's deprecated
