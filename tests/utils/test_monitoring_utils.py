import tempfile
from collections import namedtuple
from pathlib import Path

import mock
import numpy as np
import pandas as pd
import pytest
from pandas._testing import makeDataFrame

from hardpicks.utils.monitoring_utils import (
    GpuMonitor,
    append_to_file,
    monitor,
)


def get_fake_handle(i: int):
    return i


Rate = namedtuple("rate", ["gpu", "memory"])
Ram = namedtuple("Ram", ["percent"])


def get_fake_rate(handle: int):
    np.random.seed(handle)
    return Rate(gpu=np.random.rand(), memory=np.random.rand())


def get_fake_ram():
    np.random.seed(0)
    return Ram(percent=100 * np.random.rand())


@pytest.mark.parametrize("number_of_gpus", [1, 2, 4])
class TestGpuMonitor:
    @pytest.fixture()
    def gpu_monitor_and_mock_init(self, number_of_gpus):
        patch_target1 = "pynvml.nvmlInit"
        patch_target2 = "pynvml.nvmlDeviceGetCount"
        with mock.patch(patch_target1) as mock_init, mock.patch(
            patch_target2, return_value=number_of_gpus
        ):
            gpu_monitor = GpuMonitor()

        return gpu_monitor, mock_init

    @pytest.fixture()
    def gpu_monitor(self, gpu_monitor_and_mock_init):
        return gpu_monitor_and_mock_init[0]

    @pytest.fixture()
    def mock_init(self, gpu_monitor_and_mock_init):
        return gpu_monitor_and_mock_init[1]

    def test_initialization(self, mock_init):
        mock_init.assert_called()

    def test_gpu_count(self, gpu_monitor, number_of_gpus):
        assert gpu_monitor.device_count == number_of_gpus

    def test_shutdown(self, gpu_monitor):
        patch_target1 = "pynvml.nvmlShutdown"
        with mock.patch(patch_target1) as mock_shutdown:
            gpu_monitor.shutdown()
        mock_shutdown.assert_called()

    @pytest.fixture()
    def expected_sample_series(self, gpu_monitor):
        sample_series = pd.Series()
        sample_series["time"] = pd.Timestamp.now()
        sample_series["ram (%)"] = get_fake_ram().percent
        for gpu_index in range(gpu_monitor.device_count):
            handle = get_fake_handle(gpu_index)

            rate = get_fake_rate(handle)
            sample_series[f"gpu (%) (GPU[{gpu_index}])"] = rate.gpu
            sample_series[f"mem (%) (GPU[{gpu_index}])"] = rate.memory
        return sample_series

    def test_sample(self, gpu_monitor, expected_sample_series):
        patch_target1 = "pynvml.nvmlDeviceGetHandleByIndex"
        patch_target2 = "pynvml.nvmlDeviceGetUtilizationRates"
        patch_target3 = "psutil.virtual_memory"

        with mock.patch(patch_target1, side_effect=get_fake_handle), \
                mock.patch(patch_target2, side_effect=get_fake_rate), \
                mock.patch(patch_target3, side_effect=get_fake_ram):
            computed_sample_series = gpu_monitor.sample()

        expected = expected_sample_series.copy()
        assert type(expected["time"]) == pd.Timestamp
        del expected["time"]
        computed = computed_sample_series.copy()
        del computed["time"]

        pd.testing.assert_series_equal(expected, computed)


@pytest.yield_fixture()
def output_csv_file_path():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        file_path = Path(tmp_dir_str).joinpath("test_csv.csv")
        yield file_path


def test_append_to_file(output_csv_file_path):

    df = makeDataFrame().reset_index(drop=True)

    df1 = df[:20]
    df2 = df[20:]

    list_series1 = []
    for key, row in df1.iterrows():
        list_series1.append(row)
    append_to_file(output_csv_file_path, list_series1)

    read_df1 = pd.read_csv(output_csv_file_path)
    pd.testing.assert_frame_equal(read_df1, df1)

    list_series2 = []
    for key, row in df2.iterrows():
        list_series2.append(row)
    append_to_file(output_csv_file_path, list_series2)

    read_df2 = pd.read_csv(output_csv_file_path)
    pd.testing.assert_frame_equal(read_df2, df)


def test_monitor(output_csv_file_path):
    patch_target1 = "pynvml.nvmlInit"
    patch_target2 = "pynvml.nvmlDeviceGetCount"
    patch_target3 = "pynvml.nvmlShutdown"
    patch_target4 = "pynvml.nvmlDeviceGetHandleByIndex"
    patch_target5 = "pynvml.nvmlDeviceGetUtilizationRates"

    total_monitoring_interval = 1.0

    with mock.patch(patch_target1), mock.patch(
        patch_target2, return_value=2
    ), mock.patch(patch_target3), mock.patch(
        patch_target4, side_effect=get_fake_handle
    ), mock.patch(
        patch_target5, side_effect=get_fake_rate
    ):

        monitor(
            output_csv_file_path,
            sleep_time_interval=0.01,
            write_time_interval=0.1,
            total_monitoring_interval=total_monitoring_interval,
        )

    df = pd.read_csv(
        output_csv_file_path, parse_dates=["time"], infer_datetime_format=True
    )

    time_delta = df["time"].max() - df["time"].min()

    # check that the monitoring didn't take longer than expected.
    assert time_delta.seconds < 2 * total_monitoring_interval
