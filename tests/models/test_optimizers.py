import pytest
import torch

import hardpicks.models.optimizers as optim


@pytest.mark.parametrize(
    "optimizer_name", ["Adam", "AdamW", "torch.optim.SGD"],
)
def test_optimizer_getter(optimizer_name):
    model = torch.nn.Linear(100, 100)
    optimizer = optim.get_optimizer(
        optimizer_type=optimizer_name,
        optimizer_params={"lr": 0.001},
        model_params=model.parameters(),
    )
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer_name.rsplit(".")[-1] in str(type(optimizer))
