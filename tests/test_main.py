import argparse
import functools
import logging
import os
import shutil
import typing

import yaml

import deepdiff
import mlflow
import mock
import pytest

import hardpicks
import hardpicks.main
import hardpicks.train
import hardpicks.utils.file_utils


@pytest.fixture(autouse=True)
def cleanup_logging():
    """Clean up logging handlers.

    The tests in this module set global properties in the logging module which then collide with other tests!
    This method removes this problematic coupling.

    The "watched file handler" persists between tests and crashes other tests because the file no longer exits
    at the end of this module!
    """

    yield  # The yield means this fixture will run once at the end of the module.
    root = logging.getLogger()

    for handler in root.handlers:
        if type(handler) == logging.handlers.WatchedFileHandler:
            root.removeHandler(handler)


def test_setup_arg_parser():
    parser = hardpicks.main.setup_arg_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    with pytest.raises(SystemExit):
        _ = parser.parse_args()

    empty_raw_args = []
    with pytest.raises(SystemExit):
        _ = parser.parse_args(empty_raw_args)

    min_raw_args = [
        "--data=datadir", "--output=outputdir", "--config=configfile",
    ]
    for i in range(len(min_raw_args)):
        curr_raw_args = [min_raw_args[j] for j in range(len(min_raw_args)) if i != j]
        with pytest.raises(SystemExit):
            _ = parser.parse_args(curr_raw_args)

    args = parser.parse_args(min_raw_args)
    assert args.data == "datadir" and args.output == "outputdir" and args.config == "configfile"
    assert not args.disable_progressbar and not args.tensorboard_output and not args.mlflow_output

    args = parser.parse_args(min_raw_args + ["--disable-progressbar"])
    assert args.disable_progressbar


def get_dummy_config():
    return {"exp_name": "dummy_exp", "run_name": "dummy_run", "seed": None}


@pytest.fixture()
def local_dummy_config_file(tmpdir):
    dummy_config_path = os.path.join(tmpdir, "dummy.cfg")
    with open(dummy_config_path, "w") as fd:
        yaml.dump(get_dummy_config(), fd)
    return dummy_config_path


def test_prepare_experiment(tmpdir, local_dummy_config_file):
    output_dir_path = os.path.join(tmpdir, "output")
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path)
    run_name, exp_name, experiment_dir, hyper_params, config_file_backup_path = \
        hardpicks.main.prepare_experiment(
            config_file_path=local_dummy_config_file,
            output_dir_path=output_dir_path,
        )
    assert run_name == "dummy_run" and exp_name == "dummy_exp"
    assert experiment_dir == os.path.join(output_dir_path, "dummy_exp")
    assert os.path.isdir(experiment_dir)
    assert os.path.isfile(config_file_backup_path)
    assert not deepdiff.DeepDiff(get_dummy_config(), hyper_params)
    with open(config_file_backup_path, "r") as fd:
        backup_hyper_params = yaml.load(fd)
    assert not deepdiff.DeepDiff(get_dummy_config(), backup_hyper_params)


def test_prepare_loggers(tmpdir):
    # just make sure that the mlflow logger in particular has the proper run id
    mlflow_tracking_uri = os.path.join(tmpdir, "mlruns")
    os.makedirs(mlflow_tracking_uri, exist_ok=False)
    tensorboard_save_dir = os.path.join(tmpdir, "tbx")
    experiment_dir = os.path.join(tmpdir, "dummy_exp")
    os.makedirs(experiment_dir, exist_ok=False)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("dummy_exp")
    with mlflow.start_run() as run:
        mlflow_logger, _ = hardpicks.main.prepare_loggers(
            tensorboard_save_dir=tensorboard_save_dir,
            experiment_name="dummy_exp",
            experiment_dir=experiment_dir,
            run_name="dummy_run",
        )
        assert run.info.run_id == mlflow_logger.run_id
        assert os.path.isfile(os.path.join(experiment_dir, "console.log"))
        assert os.path.isdir(tensorboard_save_dir)


@pytest.fixture()
def fake_args_setup(tmpdir):
    root_examples_path = hardpicks.EXAMPLES_DIR
    smoketest_config_path = os.path.join(root_examples_path, "local", "fbp-unet-mini.yaml")
    assert os.path.isfile(smoketest_config_path)
    data_dir = hardpicks.FBP_ROOT_DATA_DIR
    output_dir = os.path.join(tmpdir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    input_args = [
        f"--data={data_dir}",
        f"--output={output_dir}",
        f"--config={smoketest_config_path}",
    ]
    return input_args, data_dir, output_dir, smoketest_config_path


def test_main_exec(fake_args_setup):
    input_args, data_dir, output_dir, smoketest_config_path = fake_args_setup
    with mock.patch("hardpicks.main.train", new=lambda *args, **kwargs: 1):
        with mock.patch("orion.client"):
            hardpicks.main.main(input_args)
    assert os.path.isdir(output_dir)
    expected_mlflow_dir = os.path.join(output_dir, "mlruns")
    assert os.path.isdir(expected_mlflow_dir)
    expected_tbx_dir = os.path.join(output_dir, "tensorboard")
    assert os.path.isdir(expected_tbx_dir)
    expected_exp_dir = os.path.join(output_dir, "fbp-unet-mini")
    assert os.path.isdir(expected_exp_dir)


def fake_rsync(src: typing.AnyStr, dst: typing.AnyStr, do_not_copy: typing.AnyStr):
    if not os.path.samefile(src, do_not_copy):
        hardpicks.utils.file_utils.rsync_folder(src, dst)


@pytest.mark.very_slow
def test_main_tmpdir_exec(fake_args_setup, tmpdir):
    input_args, data_dir, output_dir, smoketest_config_path = fake_args_setup
    temp_folder_path = os.path.join(tmpdir, "temp")
    os.makedirs(temp_folder_path, exist_ok=False)
    mlflow_folder_path = os.path.join(tmpdir, "mlruns")
    os.makedirs(mlflow_folder_path, exist_ok=False)
    input_args.extend([
        f"--tmp-folder={temp_folder_path}",
        f"--mlflow-output={mlflow_folder_path}",
    ])
    with mock.patch("hardpicks.main.train", new=lambda *args, **kwargs: 1):
        with mock.patch("hardpicks.main.create_data_module"):
            rsync = functools.partial(fake_rsync, do_not_copy=data_dir)
            with mock.patch("hardpicks.main.rsync_folder", new=rsync):
                with mock.patch("orion.client"):
                    hardpicks.main.main(input_args)
    assert os.path.isdir(output_dir)
    unexpected_mlflow_dir = os.path.join(output_dir, "mlruns")
    assert not os.path.isdir(unexpected_mlflow_dir)
    expected_tbx_dir = os.path.join(output_dir, "tensorboard")
    assert os.path.isdir(expected_tbx_dir)
    expected_exp_dir = os.path.join(output_dir, "fbp-unet-mini")
    assert os.path.isdir(expected_exp_dir)
