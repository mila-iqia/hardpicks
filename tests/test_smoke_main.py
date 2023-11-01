import logging
import os
import re
import tempfile
from pathlib import Path

import mlflow
import mock
import numpy as np
import pytest
import yaml

from hardpicks import predict
import hardpicks.main as main
from hardpicks.metrics.fbp.evaluator import FBPEvaluator
from tests import FBP_STANDALONE_SMOKE_TEST_DIR
from tests.fake_fbp_data_utilities import (
    create_fake_site_data,
    create_fake_hdf5_dataset,
)
from tests.smoke_test.fbp.smoke_test_data import (
    get_site_info,
    train_specs1,
    train_specs2,
    train_specs3,
    valid_specs,
    test_specs,
)
from tests.smoke_test.smoke_test_utils import get_directories


scope = "module"


@pytest.fixture(scope=scope, autouse=True)
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


def fake_get_site_info_array(data_dir):
    """Generate fake sites."""
    site_info_array = [
        get_site_info(train_specs1, data_dir),
        get_site_info(train_specs2, data_dir),
        get_site_info(train_specs3, data_dir),
        get_site_info(valid_specs, data_dir),
        get_site_info(test_specs, data_dir),
    ]

    return site_info_array


@pytest.fixture(scope=scope)
def temporary_directory():
    """Temp directory. tmpdir doesn't work for session scope."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope=scope)
def directories(temporary_directory):
    """Needed directories."""
    return get_directories(temporary_directory)


@pytest.fixture(scope=scope)
def main_input_args(directories):
    """Input arguments for main."""
    path_to_config_file = str(FBP_STANDALONE_SMOKE_TEST_DIR.joinpath("config.yaml"))
    data_dir = directories["data"]
    output_dir = directories["output"]
    mlflow_dir = directories["mlflow_output"]
    tb_dir = directories["tb_output"]
    input_args = [
        f"--data={data_dir }",
        f"--output={output_dir}",
        f"--config={path_to_config_file}",
        f"--mlflow-output={mlflow_dir}",
        f"--tensorboard-output={tb_dir}",
        "--disable-logger-writer",
    ]

    for specs in [train_specs1, train_specs2, train_specs3, valid_specs, test_specs]:
        hdf5_path = Path(data_dir).joinpath(f"{specs.site_name}.hdf5")
        fake_data = create_fake_site_data(specs)
        create_fake_hdf5_dataset(fake_data, hdf5_path)

    return input_args


@pytest.fixture(scope=scope)
def drive_main(main_input_args):
    """Execute the main function to generate artifacts that will be tested against."""
    mock_target = (
        "hardpicks.data.fbp.site_info.get_site_info_array"
    )
    with mock.patch(mock_target, new=fake_get_site_info_array):
        main.main(main_input_args)


@pytest.fixture(scope=scope)
def mlflow_data(drive_main):
    """Fetch the logged information in mlflow."""
    client = mlflow.tracking.MlflowClient()
    run_info = mlflow.list_run_infos("0")[0]
    run_id = run_info.run_id
    artifact_base_path = run_info.artifact_uri
    run = client.get_run(run_id)
    yield client, run, run_id, artifact_base_path

    logging.info("Cleanly terminate mlflow before tmp folders get deleted.")
    mlflow.end_run()


@pytest.fixture(scope=scope)
def checkpoint_model_path_and_mlflow_metrics(mlflow_data):
    """Extract needed information from mlflow."""
    client, run, run_id, artifact_base_path = mlflow_data
    model_path = None
    best_step = None
    checkpoint_directory_path = os.path.join(artifact_base_path, "checkpoints")
    for model_file_name in os.listdir(checkpoint_directory_path):
        if "best" in model_file_name:
            model_path = os.path.join(checkpoint_directory_path, model_file_name)
            _, best_step = parse_epoch_and_step_from_checkpoint_filename(model_path)

    mlflow_metrics_dict = get_metrics_dictionary_from_mlflow(
        client, run, run_id, best_step
    )
    return model_path, mlflow_metrics_dict


@pytest.fixture(scope=scope)
def checkpoint_model_path(checkpoint_model_path_and_mlflow_metrics):
    """Extract the checkpoint model path."""
    return checkpoint_model_path_and_mlflow_metrics[0]


@pytest.fixture(scope=scope)
def mlflow_metrics_dict(checkpoint_model_path_and_mlflow_metrics):
    """Extract the mlflow_metrics_dict."""
    return checkpoint_model_path_and_mlflow_metrics[1]


@pytest.fixture(scope=scope)
def validation_dataframe_path(mlflow_data):
    """Get the path to the validation datadump."""
    _, _, _, artifact_base_path = mlflow_data
    data_dumps_path = os.path.join(artifact_base_path, "data_dumps")
    validation_dataframe_path = None
    for filename in os.listdir(data_dumps_path):
        if "valid" in filename:
            validation_dataframe_path = os.path.join(data_dumps_path, filename)
    return validation_dataframe_path


@pytest.fixture(scope=scope)
def path_to_prediction_config(temporary_directory, checkpoint_model_path):
    """Create a predict config file on the fly."""
    path_to_config_file = str(FBP_STANDALONE_SMOKE_TEST_DIR.joinpath("config.yaml"))
    with open(path_to_config_file, "r") as stream:
        hyper_params = yaml.load(stream, Loader=yaml.FullLoader)

    hyper_params["exp_name"] = "smoke_test_prediction"
    hyper_params["model_checkpoint"] = checkpoint_model_path

    path_to_predict_config_file = os.path.join(
        temporary_directory, "predict_config.yaml"
    )
    with open(path_to_predict_config_file, "w") as stream:
        yaml.dump(hyper_params, stream)

    return path_to_predict_config_file


@pytest.fixture(scope=scope)
def predict_input_args(directories, path_to_prediction_config):
    """Create a predict inputs."""
    data_dir = directories["data"]
    output_dir = directories["output"]
    input_args = [
        f"--data={data_dir}",
        f"--output={output_dir}",
        f"--config={path_to_prediction_config}",
        "--disable-logger-writer",
    ]
    return input_args


@pytest.fixture(scope=scope)
def drive_predict(predict_input_args):
    """Execute the predict.main to generate the artifacts that will be tested against."""
    mock_target = (
        "hardpicks.data.fbp.site_info.get_site_info_array"
    )
    with mock.patch(mock_target, new=fake_get_site_info_array):
        pred_results = predict.predict(predict_input_args)
    assert pred_results is not None


@pytest.mark.slow
def test_main(mlflow_metrics_dict, validation_dataframe_path):
    """Test the main script data dump is consistent with mlflow logged results."""
    validation_evaluator = FBPEvaluator.load(validation_dataframe_path)
    evaluation_metrics_dict = get_metrics_dictionary_from_evaluator(
        "valid", validation_evaluator
    )

    for key, evaluator_value in evaluation_metrics_dict.items():
        assert key in mlflow_metrics_dict
        mlflow_value = mlflow_metrics_dict[key]
        np.testing.assert_almost_equal(mlflow_value, evaluator_value, decimal=6)


def test_predict(drive_predict, mlflow_metrics_dict, directories):
    """Test the predict.main results are consistent with mlflow."""
    output_dir = directories["output"]
    evaluator_metrics_dict = dict()
    for filename in os.listdir(output_dir):
        if "valid.pkl" in filename or "test.pkl" in filename:
            datadump_path = os.path.join(output_dir, filename)
            kind = parse_dataset_kind_from_dump_filename(datadump_path)
            evaluator = FBPEvaluator.load(datadump_path)
            evaluator_metrics_dict.update(
                get_metrics_dictionary_from_evaluator(kind, evaluator)
            )

    for metric_name in mlflow_metrics_dict.keys():
        if metric_name not in evaluator_metrics_dict:
            continue  # some metrics might be mlflow or evaluator-only; skip when so
        # ... otherwise, the resulting values should be the same!
        mlflow_value = mlflow_metrics_dict[metric_name]
        evaluator_value = evaluator_metrics_dict[metric_name]
        np.testing.assert_almost_equal(mlflow_value, evaluator_value, decimal=6)


def get_metrics_dictionary_from_evaluator(prefix, evaluator):
    """Extract all evaluator metrics with key names consistent with mlflow."""
    evaluation_dict = dict()
    for category in evaluator.get_categories():
        metrics = evaluator.summarize(category_name=category)
        for metric_name, metric_val in metrics.items():
            evaluation_dict[f"{category}/{prefix}/{metric_name}"] = metric_val
    metrics = evaluator.summarize(category_name=None)
    for metric_name, metric_val in metrics.items():
        evaluation_dict[f"{prefix}/{metric_name}"] = metric_val
    return evaluation_dict


def get_metrics_dictionary_from_mlflow(client, run, run_id, step):
    """Extract dictionary of results from mlflow."""
    mlflow_metrics_dict = dict()
    latest_metrics = run.data.metrics
    for metric_name in latest_metrics.keys():
        if "test" in metric_name:
            mlflow_metrics_dict[metric_name] = latest_metrics[metric_name]
        else:
            for metric in client.get_metric_history(run_id, metric_name):
                if metric.step == step:
                    mlflow_metrics_dict[metric.key] = metric.value
    return mlflow_metrics_dict


def parse_epoch_and_step_from_checkpoint_filename(checkpoint_filename):
    """Extract epoch and step from filename."""
    # regex explanation:
    #  (?P<...> pattern)  : <...> is the name of the group for latter retrieval; the pattern is what is matched
    # \d+ : match one or more digits.
    match = re.search(r"epoch=(?P<epoch>\d+)-step=(?P<step>\d+)", checkpoint_filename)
    epoch = int(match.group("epoch"))
    step = int(match.group("step"))
    return epoch, step


def parse_dataset_kind_from_dump_filename(dump_filename):
    """Extract the kind of dataloader the data dump was obtained from the filename."""
    # regex explanation:
    #  (?P<...> pattern)  : <...> is the name of the group for latter retrieval; the pattern is what is matched
    # [a-z]+ : match one or more lower case letters.
    match = re.search(r"_(?P<kind>[a-z]+)\.pkl", dump_filename)
    kind = match.group("kind")
    return kind
