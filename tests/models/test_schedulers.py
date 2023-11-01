import numpy as np
import pytest
import torch

import hardpicks.models.optimizers as optim
import hardpicks.models.schedulers as scheds


@pytest.fixture
def optimizer():
    model = torch.nn.Linear(100, 100)
    optimizer = optim.get_optimizer(
        optimizer_type="SGD",
        optimizer_params={"lr": 0.001},
        model_params=model.parameters(),
    )
    return optimizer


@pytest.mark.parametrize(
    "scheduler_name,scheduler_params", [
        ("ExponentialLR", {"gamma": 0.1}),
        ("StepLR", {"step_size": 1}),
    ],
)
def test_scheduler_getter(optimizer, scheduler_name, scheduler_params):
    scheduler = scheds.get_scheduler(
        scheduler_type=scheduler_name,
        scheduler_params=scheduler_params,
        optimizer=optimizer,
        max_epochs=1,
        step_count_per_epoch=10,
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)
    assert scheduler_name.rsplit(".")[-1] in str(type(scheduler))


def test_linear_warmup_cosine_annealing_only_warmup(optimizer):
    scheduler = scheds.get_scheduler(
        scheduler_type="LinearWarmupCosineAnnealingLR",
        scheduler_params={"warmup_ratio": 1.0},
        optimizer=optimizer,
        max_epochs=2,
        step_count_per_epoch=50,
    )
    assert isinstance(scheduler.get_last_lr(), list) and len(scheduler.get_last_lr()) == 1
    assert scheduler.get_last_lr()[0] == 0.0  # default (start)
    assert scheduler.warmup_steps == 100
    expected_lrs = (np.arange(1, 100).astype(np.float32) / 100) * 0.001
    for expected_lr in expected_lrs:
        scheduler.step()
        assert np.isclose(scheduler.get_lr()[0], expected_lr)
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.001  # default (end of warmup)
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.0  # end of scheduler
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.0  # past-the-end of scheduler


def test_linear_warmup_cosine_annealing(optimizer):
    scheduler = scheds.get_scheduler(
        scheduler_type="LinearWarmupCosineAnnealingLR",
        scheduler_params={"warmup_ratio": 0.5},
        optimizer=optimizer,
        max_epochs=4,
        step_count_per_epoch=50,
    )
    assert isinstance(scheduler.get_last_lr(), list) and len(scheduler.get_last_lr()) == 1
    assert scheduler.get_last_lr()[0] == 0.0  # default (start)
    print(scheduler.base_lrs)
    expected_lrs = (np.arange(1, 100).astype(np.float32) / 100) * 0.001
    for step_idx, expected_lr in enumerate(expected_lrs):
        scheduler.step()
        assert np.isclose(scheduler.get_last_lr()[0], expected_lr)
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.001  # default (end of warmup)
    for step_idx, unexpected_lr in enumerate(reversed(expected_lrs)):
        scheduler.step()
        real_lr = scheduler.get_last_lr()[0]
        assert real_lr <= 0.001  # should never go high than original LR with this scheduler...
        if step_idx < 50:
            assert np.isclose(real_lr, unexpected_lr) or real_lr >= unexpected_lr
        elif step_idx > 50:
            assert np.isclose(real_lr, unexpected_lr) or real_lr <= unexpected_lr
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.0  # end of scheduler
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.0  # past-the-end of scheduler
