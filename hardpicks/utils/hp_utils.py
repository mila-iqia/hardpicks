import copy
import logging
import mock
import os
import typing
import uuid

import deepdiff
import mlflow
import orion
import orion.client
try:
    import sigopt
    sigopt_available = True
except ImportError:
    sigopt = mock.Mock()  # if sigopt is unavailable, just run without it
    sigopt_available = False
import yaml

logger = logging.getLogger(__name__)

SIGOPT_HP_TOKEN = "_SIGOPT_"  # token used to flag the hyperparameters to be fetched via SigOpt


def check_and_log_hp(names, hps, allow_extra=True):  # pragma: no cover
    """Check and log hyper-parameters.

    Args:
        names (list): names of all expected hyper parameters
        hps (dict): all hyper-parameters from the config file
        allow_extra (bool): Can have more hyper-parameters than explicitly stated
    """
    check_hp(names, hps, allow_extra=allow_extra)
    log_hp(names, hps)


def check_hp(names, hps, allow_extra=True):
    """Check if required hyper-parameters are all present.

    Args:
        names (list): names of all expected hyper parameters
        hps (dict): all hyper-parameters from the config file
        allow_extra (bool): Can have more hyper-parameters than explicitly stated
    """
    missing = set()
    for name in names:
        if name not in hps:
            missing.add(name)
    extra = hps.keys() - names
    err_msg = ""
    if len(missing) > 0:
        err_msg += f"please add the missing hyper-parameters:\n\t{missing}\n"
    if len(extra) > 0 and not allow_extra:
        err_msg += f"please remove the extra hyper-parameters:\n\t{extra}\n"
    if err_msg:
        err_msg = f"HYPER PARAMETER CONFIG ERROR!\n{err_msg}"
        logger.error(err_msg)
        raise ValueError(err_msg)


def log_hp(names, hps, name_prefix=None):  # pragma: no cover
    """Log the hyper-parameters.

    Args:
        names (list): list with names of hyper parameters to log
        hps (dict): all hyper-parameters from the config file
        name_prefix (optional, str): the prefix to prepend to variable names
    """
    for name in sorted(names):
        val = hps[name]
        if isinstance(val, dict):
            assert all([isinstance(key, str) for key in val.keys()]), \
                "cannot log hyperparam dictionaries unless all keys are strings!"
            new_prefix = name if not name_prefix else name_prefix + "/" + name
            log_hp(names=list(val.keys()), hps=val, name_prefix=new_prefix)
        else:
            hp_name = name if not name_prefix else name_prefix + "/" + name
            mlflow.log_param(hp_name, val)
            logger.info(f"hyperparameter '{hp_name}' => {str(val)}")


def load_mlflow_run_info(path: typing.AnyStr):
    """Load the mlflow run id stored in a given file path."""
    with open(path, "r") as stream:
        info = yaml.load(stream, Loader=yaml.FullLoader)
    return info


def write_mlflow_run_info(path: typing.AnyStr):
    """Store the (currently active) mlflow run's id at the specified file path."""
    mlflow_run = mlflow.active_run()
    experiment = mlflow.get_experiment(mlflow_run.info.experiment_id)
    info = {
        "experiment_id": mlflow_run.info.experiment_id,
        "experiment_name": experiment.name,
        "experiment_tags": experiment.tags,
        "run_id": mlflow_run.info.run_id,
        "start_time": mlflow_run.info.start_time,
        "user_id": mlflow_run.info.user_id,
        "tracking_uri": mlflow.get_tracking_uri(),
        "artifact_uri": mlflow.get_artifact_uri(),
    }
    with open(path, "w") as stream:
        yaml.dump(info, stream)


def get_array_from_input_that_could_be_a_string(
    array_in_unknown_format: typing.Union[str, list],
) -> list:
    """Function to turn string representation of a list into a list."""
    if type(array_in_unknown_format) == list:
        return array_in_unknown_format

    elif type(array_in_unknown_format) == str:
        output_array = eval(array_in_unknown_format, None, None)
        assert type(output_array) == list, "something is wrong with input."
        return output_array


def get_run_name(
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
) -> typing.AnyStr:
    """Returns the run name for the current experiment, given the current app state + hyperparams."""
    assert isinstance(hyper_params, dict)
    # if sigopt is required, set it up and connect to the experiment (should be created via CLI)
    need_sigopt = check_config_contains_sigopt_tokens(hyper_params=hyper_params)
    if need_sigopt:
        assert sigopt_available, "sigopt unavailable (make sure it is installed in the right env!)"
        assert not orion.client.cli.IS_ORION_ON, "we should not be mixing orion + sigopt for run ids"
        assert "run_name" not in hyper_params or hyper_params["run_name"] == SIGOPT_HP_TOKEN, \
            "run name should be provided by sigopt, not assigned by the user/config"
        run_name = sigopt.get_run_id()  # we should already be in a run started via CLI!
    else:
        default_run_name = str(uuid.uuid4())  # will be used in case we've got no other ID...
        if orion.client.cli.IS_ORION_ON and "run_name" not in hyper_params:
            run_name = str(os.getenv("ORION_TRIAL_ID", default=default_run_name))
        else:
            run_name = str(hyper_params.get("run_name", default_run_name))
    return run_name


def replace_placeholders(
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
) -> typing.Dict[typing.AnyStr, typing.Any]:
    """Replaces placeholder hyperparameters and returns a copy of the updated dict."""
    assert isinstance(hyper_params, dict)
    # if sigopt is required, set it up and connect to the experiment (should be created via CLI)
    need_sigopt = check_config_contains_sigopt_tokens(hyper_params=hyper_params)
    if need_sigopt:
        assert sigopt_available, "sigopt unavailable (make sure it is installed in the right env!)"
        assert not orion.client.cli.IS_ORION_ON, "we should not be mixing orion + sigopt for run ids"
        assert "run_name" not in hyper_params or hyper_params["run_name"] == SIGOPT_HP_TOKEN, \
            "run name should be provided by sigopt, not assigned by the user/config"
        run_name = sigopt.get_run_id()  # we should already be in a run started via CLI!
        hyper_params = copy.deepcopy(hyper_params)
        hyper_params["run_name"] = run_name
        # time to replace all the dummy placeholders by the sigopt-suggest values...
        hyper_params = find_and_replace_sigopt_hyperparameters(
            hyper_params=hyper_params,
            log_non_sigopt_hps_as_metadata=True,  # this will log all non-sigopt-related hyperparams
            sigopt_param_name_spacer=hyper_params.get("sigopt_param_name_spacer", "_")
        )
    # otherwise, without sigopt, we've got nothing to replace inside the hparam dict...
    return hyper_params


def check_config_contains_sigopt_tokens(
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
) -> bool:
    """Returns whether the provided hyperparameter config contains any sigopt token or not."""
    for hp_key, hp_val in hyper_params.items():
        if isinstance(hp_val, str) and hp_val == SIGOPT_HP_TOKEN:
            return True
        elif isinstance(hp_val, dict) and all([isinstance(k, str) for k in hp_val.keys()]):
            if check_config_contains_sigopt_tokens(hyper_params=hp_val):
                return True
    return False


def set_sigopt_global_run_context(run) -> None:
    """Sets the SigOpt global run context directly so that experiments don't have to fetch it."""
    assert sigopt_available, "sigopt unavailable (make sure it is installed in the right env!)"
    assert isinstance(run, (sigopt.run_context.TrainingRun, sigopt.run_context.RunContext))
    if isinstance(run, sigopt.run_context.TrainingRun):
        run = sigopt.run_context.RunContext(sigopt.interface.get_connection(), run)
    sigopt.run_context.global_run_context.set_run_context(run)


def find_and_replace_sigopt_hyperparameters(
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    sigopt_param_prefix: typing.AnyStr = "",  # used for parameters embedded into parent dicts
    log_non_sigopt_hps_as_metadata: bool = False,
    sigopt_param_name_spacer: str = "_",
) -> typing.Dict[typing.AnyStr, typing.Any]:
    """Finds and fills the value of hyperparameters that should be provided via SigOpt.

    This function will recursively enter into dictionaries and try to find/replace other keys. The
    returned dictionary is a copy of the original with the same keys, but in which the sigopt tokens
    will have been substituted by their sigopt-suggested values.

    The function can also (optionally) log all non-sigopt-provided values as metadata.
    """
    assert sigopt_available, "sigopt unavailable (make sure it is installed in the right env!)"
    already_copied = False
    for hp_key, hp_val in hyper_params.items():
        if isinstance(hp_val, str) and hp_val == SIGOPT_HP_TOKEN:
            expected_sigopt_hp_key = sigopt_param_prefix + hp_key
            if expected_sigopt_hp_key == "run_name":
                hp_val = sigopt.get_run_id()
                assert hp_val is not None
            else:
                assert expected_sigopt_hp_key in sigopt.params, \
                    f"sigopt is missing parameter with key '{expected_sigopt_hp_key}'"
                hp_val = sigopt.params[expected_sigopt_hp_key]
            if not already_copied:
                hyper_params = copy.deepcopy(hyper_params)  # makes sure we don't break other stuff
                already_copied = True
            hyper_params[hp_key] = hp_val
        elif isinstance(hp_val, dict) and all([isinstance(k, str) for k in hp_val.keys()]):
            new_subdict = find_and_replace_sigopt_hyperparameters(
                hyper_params=hp_val,
                sigopt_param_prefix=sigopt_param_prefix + hp_key + sigopt_param_name_spacer,
                sigopt_param_name_spacer=sigopt_param_name_spacer,
            )
            if not already_copied and deepdiff.DeepDiff(new_subdict, hp_val):
                hyper_params = copy.deepcopy(hyper_params)  # makes sure we don't break other stuff
                already_copied = True
            hyper_params[hp_key] = new_subdict
        elif log_non_sigopt_hps_as_metadata:
            sigopt.log_metadata(
                key=sigopt_param_prefix + hp_key,
                value=hp_val,
            )
    return hyper_params


def log_model_info(
    model: typing.Any,
) -> None:
    """Logs the model info (metadata only) to whatever logger is currently available."""
    # first, let's use the simplest logger, the terminal itself:
    if hasattr(model, "summary") and callable(model.summary):
        info = model.summary()  # if there's a rich summary for the model, log it!
    else:
        info = type(model).__name__  # otherwise, we'll fall back to the type name only
    logger.info(f"model info:\n{info}\n")
    # if sigopt is available, let's log it there as well
    sigopt.log_model(info)
    # TODO: add tensorboard? mlflow? others?


def log_data_module_info(
    module: typing.Any,
) -> None:
    """Logs the data module info (metadata only) to whatever logger is currently available."""
    # first, let's use the simplest logger, the terminal itself:
    if hasattr(module, "summary") and callable(module.summary):
        info = module.summary()  # if there's a rich summary for the model, log it!
    else:
        info = type(module).__name__  # otherwise, we'll fall back to the type name only
    logger.info(f"data module info:\n{info}\n")
    # if sigopt is available, let's log it there as well
    sigopt.log_dataset(info)
    # TODO: add tensorboard? mlflow? others?


def log_run_failure(
    err: Exception,
) -> None:
    """Logs the message tied to the error that failed this run."""
    # first: let's use the simplest logger, the terminal itself:
    logger.error(f"Caught {type(err)}: {str(err)}")
    # if sigopt is available, log the error message as a metadata entry as well
    sigopt.log_failure()
    err_msg = str(err)
    max_str_len = min(len(err_msg), 500)
    sigopt.log_metadata("exception", str(err)[-max_str_len:])


def log_metric(
    metric_name: typing.AnyStr,
    metric_val: float,
) -> None:
    """Logs a metric name-value pair to the background sigopt/mlflow loggers.

    Note: this is not typically used directly during training with PyTorch-Lightning, as calling
    `self.log(...)` during an experiment will likely be a more direct solution. However, this call
    may be necessary if we want to log metrics OUTSIDE the training scope, e.g. when we are
    reporting metrics on a final test dataset, or on a downstream task.
    """
    sigopt.log_metric(name=metric_name, value=metric_val)
    mlflow.log_metric(key=metric_name, value=metric_val)
