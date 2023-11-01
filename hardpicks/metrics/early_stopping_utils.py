import typing

from hardpicks.metrics.fbp.evaluator import (
    HIGHER_IS_BETTER_METRICS as FBP_HIGHER_IS_BETTER_METRICS,
    LOWER_IS_BETTER_METRICS as FBP_LOWER_IS_BETTER_METRICS,
)

# remove potential duplicates by using sets.
GLOBAL_HIGHER_IS_BETTER_METRICS = list(
    set(
        FBP_HIGHER_IS_BETTER_METRICS
    )
)

GLOBAL_LOWER_IS_BETTER_METRICS = list(
    set(
        FBP_LOWER_IS_BETTER_METRICS
    )
)


def get_early_stopping_metric_mode(early_stopping_metric: str) -> str:
    """Get mode for early stopping.

    The early stopping callback needs to know if the metric should
    increase or decrease to improve.

    Args:
        metric: the metric string

    Returns:
        mode: the pytorch lightning early stopping callback mode.
    """
    metric_was_found = False
    mode = "none"

    # In max mode, training will stop when the quantity
    # monitored has stopped increasing
    for metric in GLOBAL_HIGHER_IS_BETTER_METRICS:
        if metric in early_stopping_metric:
            metric_was_found = True
            mode = "max"

    # In min mode, training will stop when the quantity
    # monitored has stopped decreasing
    for metric in GLOBAL_LOWER_IS_BETTER_METRICS:
        if metric in early_stopping_metric:
            metric_was_found = True
            mode = "min"
    assert metric_was_found, "The early stopping metric was not found. Review code."
    return mode


def get_name_and_sign_of_orion_optimization_objective(
    early_stopping_metric: str,
    mode: typing.Optional[typing.AnyStr] = None,
) -> typing.Tuple[str, int]:
    """Names and signs.

    The Orion optimizer seeks to minimize an objective. Some metrics must be maximized,
    and others must be minimized. This function returns what is needed to align with Orion.

    Args:
        early_stopping_metric: name of the early stop metric, as passed in the input config file.
        mode: optimization mode for the metric ('min', 'max', or None for auto-detect).

    Returns:
        optimization_objective_name: A proper name for what Orion will be optimizing
        optimization_sign: premultiplicative factor (+/- 1) to make sure Orion tries to minimize an objective.
    """
    if mode is None:
        mode = get_early_stopping_metric_mode(early_stopping_metric)

    if mode == "max":
        # The metric must be maximized. Correspondingly, Orion will minimize minus x metric.
        optimization_objective_name = f"minus_{early_stopping_metric}"
        optimization_sign = -1
    elif mode == "min":
        # The metric must be minimized; this is already aligned with what Orion will optimize.
        optimization_objective_name = early_stopping_metric
        optimization_sign = 1
    else:
        raise ValueError("The mode for this early_stopping_metric is unknown")
    return optimization_objective_name, optimization_sign
