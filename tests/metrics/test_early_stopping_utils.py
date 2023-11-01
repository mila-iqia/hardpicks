import pytest

from hardpicks.metrics.early_stopping_utils import get_early_stopping_metric_mode, \
    get_name_and_sign_of_orion_optimization_objective
from hardpicks.metrics.fbp.evaluator import HIGHER_IS_BETTER_METRICS, LOWER_IS_BETTER_METRICS


def test_get_early_stopping_metric_mode():

    for higher_is_better in HIGHER_IS_BETTER_METRICS:
        early_stopping_metric = f"val_{higher_is_better}_blahblah"
        mode = get_early_stopping_metric_mode(early_stopping_metric)
        assert mode == 'max'

    for lower_is_better in LOWER_IS_BETTER_METRICS:
        early_stopping_metric = f"val_{lower_is_better}_blahblah"
        mode = get_early_stopping_metric_mode(early_stopping_metric)
        assert mode == 'min'

    early_stopping_metric = "I_dont_exist"

    with pytest.raises(AssertionError):
        _ = get_early_stopping_metric_mode(early_stopping_metric)


def test_get_name_and_sign_of_orion_optimization_objective():
    for higher_is_better in HIGHER_IS_BETTER_METRICS:
        early_stopping_metric = f"val_{higher_is_better}_blahblah"
        expected_optimization_objective_name = "minus_" + early_stopping_metric
        expected_optimization_sign = -1

        computed_optimization_objective_name, computed_optimization_sign = \
            get_name_and_sign_of_orion_optimization_objective(early_stopping_metric)

        assert computed_optimization_objective_name == expected_optimization_objective_name
        assert computed_optimization_sign == expected_optimization_sign

    for lower_is_better in LOWER_IS_BETTER_METRICS:
        early_stopping_metric = f"val_{lower_is_better}_blahblah"
        expected_optimization_objective_name = early_stopping_metric
        expected_optimization_sign = 1

        computed_optimization_objective_name, computed_optimization_sign = \
            get_name_and_sign_of_orion_optimization_objective(early_stopping_metric)
        assert computed_optimization_objective_name == expected_optimization_objective_name
        assert computed_optimization_sign == expected_optimization_sign
