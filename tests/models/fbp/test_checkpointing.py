import mock
import pytest
import torch
import yaml
from torch import optim

from hardpicks.models.losses import get_loss_function
from hardpicks.models.model_loader import load_model
from hardpicks.utils.reproducibility_utils import set_seed
from tests import FBP_STANDALONE_SMOKE_TEST_DIR


@pytest.fixture
def hyper_params(encoder_name):
    file_name = None
    if encoder_name == "vanilla":
        file_name = "config.yaml"
    elif encoder_name == "resnet":
        file_name = "config_resnet.yaml"
    elif encoder_name == "efficientnet":
        file_name = "config_efficientnet.yaml"

    path_to_config_file = str(FBP_STANDALONE_SMOKE_TEST_DIR.joinpath(file_name))
    with open(path_to_config_file, "r") as stream:
        hyper_params = yaml.load(stream, Loader=yaml.FullLoader)
    return hyper_params


@pytest.fixture
def criterion(hyper_params):
    loss_type = hyper_params["loss_type"]
    loss_params = hyper_params["loss_params"]
    segm_class_count = hyper_params["segm_class_count"]
    if segm_class_count == 1:
        loss_mode = "binary"
    else:
        loss_mode = "multiclass"
    criterion = get_loss_function(loss_type, loss_mode, loss_params, -1)
    return criterion


@pytest.fixture
def seed():
    return 2342


@pytest.fixture
def fake_inputs_and_labels(hyper_params):
    segm_class_count = hyper_params["segm_class_count"]
    # it is strange that the inputs must have requires_grad=True. Without it the test fails.
    fake_input = torch.rand(16, 4, 32, 32, requires_grad=False)
    fake_labels = torch.randint(0, segm_class_count - 1, (16, 32, 32))
    return fake_input, fake_labels


@pytest.fixture
def model_with_checkpointing(hyper_params, seed):
    try:
        import fairscale.nn  # noqa
    except ImportError:
        pytest.skip("skipping model checkpointing test since fairscale is not installed")
    hyper_params_with_checkpointing = dict(hyper_params)
    hyper_params_with_checkpointing["use_checkpointing"] = True
    set_seed(seed, set_deterministic=True, set_benchmark=False)
    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        model_with_checkpointing = load_model(hyper_params_with_checkpointing)
    model_with_checkpointing.zero_grad()
    return model_with_checkpointing


@pytest.fixture
def model_without_checkpointing(hyper_params, seed):
    hyper_params_with_checkpointing = dict(hyper_params)
    hyper_params_with_checkpointing["use_checkpointing"] = False
    set_seed(seed, set_deterministic=True, set_benchmark=False)
    with mock.patch("hardpicks.utils.hp_utils.log_hp"):
        model_without_checkpointing = load_model(hyper_params_with_checkpointing)
    model_without_checkpointing.zero_grad()
    return model_without_checkpointing


@pytest.mark.parametrize("encoder_name", ["vanilla", "resnet", "efficientnet"])
def test_checkpointing_gradients(
    fake_inputs_and_labels,
    criterion,
    model_with_checkpointing,
    model_without_checkpointing,
    seed
):
    fake_input, fake_labels = fake_inputs_and_labels
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    for model in [model_without_checkpointing, model_with_checkpointing]:
        set_seed(seed, set_deterministic=True, set_benchmark=False)
        # run a gradient calculation step for each model
        model = model.to(device)
        fake_logits = model(fake_input.to(device))
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer.zero_grad()
        loss = criterion(fake_logits, fake_labels.to(device))
        loss.backward()
        optimizer.step()

    # make sure all the gradients are the same with and without checkpointing
    list_parameters1 = list(model_with_checkpointing.parameters())
    list_of_parameters2 = list(model_without_checkpointing.parameters())

    assert len(list_parameters1) == len(
        list_of_parameters2
    ), "the number of parameters are not the same"

    number_of_non_none = 0
    for p1, p2 in zip(list_parameters1, list_of_parameters2):
        if p1.grad is None:
            assert p2.grad is None
        else:
            assert torch.eq(p1.grad, p2.grad).all(), " the gradients are not the same"
            number_of_non_none += 1

    assert number_of_non_none > 0
