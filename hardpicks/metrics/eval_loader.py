"""Metrics evaluator loader & other utilities."""

import logging
import typing

import hardpicks.utils.hp_utils
from hardpicks.metrics.base import EvaluatorBase, NoneEvaluator
from hardpicks.metrics.fbp.evaluator import FBPEvaluator

logger = logging.getLogger(__name__)

SUPPORTED_EVALUATOR_PAIRS = [
    ("NoneEvaluator", NoneEvaluator),
    ("FBPEvaluator", FBPEvaluator),
]


def _get_matching_evaluator_class(
    eval_name: typing.AnyStr,
) -> typing.Optional[typing.Type]:
    """Returns the type that matches a supported eval, or ``None`` if no match is found."""
    for potential_eval_name, eval_class in SUPPORTED_EVALUATOR_PAIRS:
        if potential_eval_name == eval_name:
            return eval_class
    return None


def get_evaluator(
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
) -> EvaluatorBase:
    """Instantiate a an evaluator object by name, given a parameter dict.

    The evaluator will compare prediction results and targets during an epoch and provide
    an interface to summarize the comparisons into useful metrics at the end of the epoch.

    Args:
        hyper_params: hyper parameters dictionary loaded from the experiment configuation file.

    Returns:
        An evaluator object.
    """
    if "eval_type" not in hyper_params:  # quick-exit if we do not define an evaluator at all
        return NoneEvaluator(hyper_params)

    hardpicks.utils.hp_utils.log_hp(
        names=["eval_type"],
        hps=hyper_params,
    )
    eval_type = hyper_params["eval_type"]
    eval_class = _get_matching_evaluator_class(eval_type)
    assert eval_class is not None, f"unsupported evaluator type: {eval_type}"
    eval = eval_class(hyper_params)
    logger.info('eval info:\n' + str(eval) + '\n')
    return eval
