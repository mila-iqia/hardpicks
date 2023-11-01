import itertools
from collections import namedtuple
from typing import List

import numpy as np
import pytest
import torch.random

from hardpicks.models.losses import get_loss_function

Prediction = namedtuple("prediction", ["ground_truth_label", "softmax_probabilities", "logits"])


def get_softmax_for_testing(logits: np.array) -> np.array:
    assert len(logits.shape) == 4
    batch_size, number_of_classes, width, height = logits.shape

    # compute the softmax explicitly
    exp_logits = np.exp(logits)
    denominator = exp_logits.sum(axis=1)[:, np.newaxis, :, :].repeat(
        number_of_classes, axis=1
    )
    softmax = exp_logits / denominator
    return softmax


def explicit_cross_entropy_for_testing(list_predictions: List[Prediction]):

    all_ground_truth_probabilities = []
    for prediction in list_predictions:
        index = prediction.ground_truth_label
        probability = prediction.softmax_probabilities[index]
        all_ground_truth_probabilities.append(probability)

    loss = -np.log(all_ground_truth_probabilities).mean()
    return torch.tensor(loss)


def explicit_focal_loss_for_testing(list_predictions: List[Prediction], number_of_classes: int, gamma: float):
    """
    The explicit implementation below, which leads to passing tests, shows more transparently what the FocalLoss
    is computing. Basically, it is iterating over all classes and computes the focal loss for the binary problem
    "ground truth is this class" vs "ground truth is not this class". This is not the most natural generalization
    of focal loss to the multiclass problem.

    .. math::
        N : number of examples
        C: total number of classes
        gamma: the gamma parameter of the focal loss
        i : instance index
        pi_c : predicted probability of the class c, computed using a softmax
        qi_c : probability that ground truth is class c (either 1 or 0).

        Correct Focal Loss = -\frac{1}{N} \sum_{c} \sum_{i=1}^{N}  qi_c (1- pi_c)^gamma log(pi_c)

    What SMP implements, however, is

    .. math::
        ~pi_c: predicted probability of class c, computed uniquely from the c component of the logits using a sigmoid

        SMP Focal Loss = - \frac{1}{N} \sum_{c}  qi_c (1- ~pi_c)^gamma log(~pi_c) + (1-qi_c) (~pi_c)^gamma log(1-~pi_c)
    """  # noqa

    list_class_losses = []

    for cls in range(number_of_classes):
        list_class_values = []

        for prediction in list_predictions:

            gt = prediction.ground_truth_label
            logits = prediction.logits

            class_logit = logits[cls]

            # computing the probability for this class, using a simple sigmoid, ignoring
            # all other logits
            p = np.exp(class_logit) / (1. + np.exp(class_logit))

            if cls == gt:
                q = 1.0
            else:
                q = 0.0
            # binary crossentropy-like term, with the focal loss weight terms.
            loss = -q * (1. - p) ** gamma * np.log(p) - (1. - q) * p ** gamma * np.log(1 - p)
            list_class_values.append(loss)

        class_loss = np.mean(list_class_values)

        list_class_losses.append(class_loss)

    loss = np.sum(list_class_losses)

    return torch.tensor(loss)


def get_valid_predictions(
    torch_prediction_logits: torch.tensor, torch_targets: torch.tensor
) -> torch.tensor:
    logits = torch_prediction_logits.detach().numpy()
    targets = torch_targets.detach().numpy()

    assert len(logits.shape) == 4
    batch_size, number_of_classes, width, height = logits.shape

    assert targets.shape[0] == batch_size
    assert targets.shape[1] == width
    assert targets.shape[2] == height

    # compute the softmax explicitly
    softmax = get_softmax_for_testing(logits)
    list_predictions = []
    for batch_index in range(batch_size):
        for width_index in range(width):
            for height_index in range(height):

                ground_truth_label = targets[batch_index, width_index, height_index]
                if ground_truth_label == -1:
                    # don't care index: move on
                    continue
                probabilities = softmax[batch_index, :, width_index, height_index]
                lg = logits[batch_index, :, width_index, height_index]

                prediction = Prediction(ground_truth_label=ground_truth_label,
                                        softmax_probabilities=probabilities,
                                        logits=lg)
                list_predictions.append(prediction)
    return list_predictions


def explicit_dice_loss_per_class_for_testing(
    torch_prediction_logits: torch.tensor, torch_targets: torch.tensor
) -> torch.tensor:
    logits = torch_prediction_logits.detach().numpy()
    targets = torch_targets.detach().numpy()

    assert len(logits.shape) == 4
    batch_size, number_of_classes, width, height = logits.shape

    assert targets.shape[0] == batch_size
    assert targets.shape[1] == width
    assert targets.shape[2] == height

    # compute the softmax explicitly
    softmax = get_softmax_for_testing(logits)

    # Zero out the softmax when the ground truth is "don't care".
    # these pixels should not count towards the loss.
    target_is_dont_care_index_mask = np.where(targets == -1)
    batch_indices, width_indices, height_indices = target_is_dont_care_index_mask
    softmax[batch_indices, :, width_indices, height_indices] = 0.0

    list_class_losses = []
    for class_index in range(number_of_classes):
        target_is_class_mask = np.where(targets == class_index)

        number_of_class_instances = np.sum(targets == class_index)

        class_softmax = softmax[:, class_index, :, :]
        class_softmax_when_target_is_class = class_softmax[target_is_class_mask]

        intersection = np.sum(class_softmax_when_target_is_class)
        cardinality = np.sum(class_softmax) + number_of_class_instances

        class_loss = 1.0 - 2.0 * intersection / cardinality
        list_class_losses.append(class_loss)

    return list_class_losses


def get_loss_from_class_losses(loss_mode, list_class_losses):
    if loss_mode == "binary":
        # This is binary mode; we only care about the last class
        loss = list_class_losses[-1]
    else:
        # This is multiclass mode; average over all classes
        loss = np.mean(list_class_losses)
    return torch.tensor([loss])


@pytest.fixture()
def batch_size():
    return 8


@pytest.fixture()
def height():
    return 17


@pytest.fixture()
def width():
    return 9


@pytest.fixture()
def number_of_channels(number_of_segmentation_classes):
    if number_of_segmentation_classes <= 2:
        return 2
    else:
        return number_of_segmentation_classes


@pytest.fixture()
def targets(batch_size, number_of_channels, height, width):
    torch.random.manual_seed(23423)
    return torch.randint(-1, number_of_channels, [batch_size, height, width])


@pytest.fixture()
def prediction_logits(batch_size, number_of_channels, height, width):
    torch.random.manual_seed(4323452)
    logits = 5 * torch.rand([batch_size, number_of_channels, height, width]) - 10.0

    # change the logits so that the class zero logit is zero; this is to account for the
    # fact that the softmax is computed differently for the binary and multiclass cases,
    # namely using a sigmoid instead of a softmax.
    logits[:, 0, :, :] = 0.0

    return logits


@pytest.fixture()
def loss_mode(number_of_segmentation_classes):
    if number_of_segmentation_classes < 2:
        return "binary"
    else:
        return "multiclass"


@pytest.mark.parametrize("number_of_segmentation_classes", [1, 2, 3])
def test_dice_loss(number_of_segmentation_classes, loss_mode, prediction_logits, targets):
    loss_type = "dice"
    loss_params = {}

    loss_function = get_loss_function(loss_type, loss_mode, loss_params, ignore_index=-1)

    list_class_losses = explicit_dice_loss_per_class_for_testing(
        prediction_logits, targets
    )

    expected_loss = get_loss_from_class_losses(loss_mode, list_class_losses)

    computed_loss = loss_function(prediction_logits, targets)

    np.testing.assert_almost_equal(computed_loss.item(), expected_loss.item(), decimal=6)


@pytest.mark.parametrize("number_of_segmentation_classes", [1, 2, 3])
def test_cross_entropy(number_of_segmentation_classes, loss_mode, prediction_logits, targets):
    loss_type = "crossentropy"
    loss_params = {}

    loss_function = get_loss_function(loss_type, loss_mode, loss_params, ignore_index=-1)

    list_predictions = get_valid_predictions(prediction_logits, targets)

    expected_loss = explicit_cross_entropy_for_testing(list_predictions)

    computed_loss = loss_function(prediction_logits, targets)

    # we're taking logs and stuff. We can be lenient on the decimal.
    np.testing.assert_almost_equal(computed_loss.item(), expected_loss.item(), decimal=5)


@pytest.mark.parametrize("number_of_segmentation_classes, gamma", itertools.product([2, 3], [1.5, 2., 2.5]))
def test_focal_loss(number_of_segmentation_classes, gamma, loss_mode, prediction_logits, targets):
    loss_type = "focal"
    loss_params = {'gamma': gamma}

    loss_function = get_loss_function(loss_type, loss_mode, loss_params, ignore_index=-1)

    list_predictions = get_valid_predictions(prediction_logits, targets)

    expected_loss = explicit_focal_loss_for_testing(list_predictions, number_of_segmentation_classes, gamma)

    computed_loss = loss_function(prediction_logits, targets)

    # we're taking logs and stuff. We can be lenient on the decimal.
    np.testing.assert_almost_equal(computed_loss.item(), expected_loss.item(), decimal=5)
