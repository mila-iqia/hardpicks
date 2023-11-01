import deepdiff
import mock
import numpy as np
import pytest

import hardpicks.utils.hp_utils as hp_utils


def test_check_hp__all_params_are_there():
    """Test that the method check_hp works as expected for valid input data."""
    names = ['a', 'b']
    hps = {'a': 0, 'b': 1}
    hp_utils.check_hp(names, hps)


def test_check_hp__param_is_missing():
    """Test that the method check_hp raises error on invalid input data."""
    names = ['a', 'b']
    hps = {'a': 0}
    with pytest.raises(ValueError):
        hp_utils.check_hp(names, hps)


def test_check_hp__extra_param_allowed():
    """Test that the method check_hp works for valid input data with extra values."""
    names = ['a']
    hps = {'a': 0, 'b': 1}
    hp_utils.check_hp(names, hps)


def test_check_hp__extra_param_not_allowed():
    """Test that the method check_hp raises error with stringent flag."""
    names = ['a']
    hps = {'a': 0, 'b': 1}
    with pytest.raises(ValueError):
        hp_utils.check_hp(names, hps, allow_extra=False)


def test_check_and_log_hp__with_subdict():
    """Checks whether subdictionaries of parameters are parsed and logged properly."""
    names = ["a", "b", "c"]
    hps = {
        "a": {"potato": 1, "banana": {"something1": 1, "something2": 2}},
        "b": {},  # this one will not generate a logging call as it is empty
        "c": 1,
        "d": None,
    }
    with mock.patch("mlflow.log_param") as fake_logger:
        hp_utils.check_and_log_hp(names, hps)
        assert len(fake_logger.call_args_list) == 4
        logged_hps = {args[0][0]: args[0][1] for args in fake_logger.call_args_list}
        assert "a/potato" in logged_hps and logged_hps["a/potato"] == 1
        assert "a/banana/something1" in logged_hps and logged_hps["a/banana/something1"] == 1
        assert "a/banana/something2" in logged_hps and logged_hps["a/banana/something2"] == 2
        assert "c" in logged_hps and logged_hps["c"] == 1


def test_get_array_from_input_that_could_be_a_string():

    expected_list = [64, 32, 16]
    input_list = [64, 32, 16]
    input_string = "[64, 32, 16]"

    for input in [input_list, input_string]:
        computed_list = hp_utils.get_array_from_input_that_could_be_a_string(input)
        np.testing.assert_array_equal(expected_list, computed_list)


def test_check_config_contains_sigopt_tokens_and_replace_them():
    try:
        import sigopt  # noqa
    except ImportError:
        pytest.skip("skipping sigopt token replacement test since sigopt is not installed")
    fake_config = {
        "something": [1, 2, 3],
        "something else": {"hello": "hi", "potato": 123},
        "nananana": None,
        "run_name": "something",
    }
    assert not hp_utils.check_config_contains_sigopt_tokens(fake_config)
    fake_config["something else again"] = hp_utils.SIGOPT_HP_TOKEN
    assert hp_utils.check_config_contains_sigopt_tokens(fake_config)
    fake_config["something else again"] = {}
    assert not hp_utils.check_config_contains_sigopt_tokens(fake_config)
    fake_config["something else again"]["new"] = hp_utils.SIGOPT_HP_TOKEN
    assert hp_utils.check_config_contains_sigopt_tokens(fake_config)
    with mock.patch("hardpicks.utils.hp_utils.sigopt") as fake_sigopt:
        fake_sigopt.params = {"something else again_new": 13}
        new_config = hp_utils.find_and_replace_sigopt_hyperparameters(fake_config)
        assert new_config["something else again"]["new"] == 13
    assert deepdiff.DeepDiff(fake_config, new_config)


def test_replace_placeholders_and_return_run_name():
    try:
        import sigopt  # noqa
    except ImportError:
        pytest.skip("skipping sigopt token replacement test since sigopt is not installed")
    fake_config = {
        "something": [1, 2, 3],
        "something else": {"hello": "hi", "potato": 123},
        "nananana": None,
        "run_name": "potato",
    }
    run_name = hp_utils.get_run_name(fake_config)
    updt_config = hp_utils.replace_placeholders(fake_config)
    assert run_name == "potato" and not deepdiff.DeepDiff(fake_config, updt_config)
    fake_config["nananana"] = hp_utils.SIGOPT_HP_TOKEN
    fake_config["run_name"] = hp_utils.SIGOPT_HP_TOKEN
    with mock.patch("hardpicks.utils.hp_utils.sigopt_available", new=True):
        with pytest.raises(AssertionError):
            _ = hp_utils.replace_placeholders(fake_config)
        del fake_config["run_name"]
        with mock.patch("hardpicks.utils.hp_utils.sigopt") as fake_sigopt:
            fake_sigopt.params = {"nananana": 42}
            run_name = hp_utils.get_run_name(fake_config)
            updt_config = hp_utils.replace_placeholders(fake_config)
        assert isinstance(run_name, mock.Mock)
        assert deepdiff.DeepDiff(fake_config, updt_config)
        assert updt_config["nananana"] == 42
