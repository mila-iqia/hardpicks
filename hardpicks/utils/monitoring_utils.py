import time
from pathlib import Path
from typing import List

import psutil
import pynvml
import pandas as pd


_eight_hours = 8 * 60 * 60


class GpuMonitor:
    """Class to monitor GPUs.

    This class assumes that the NVIDIA tool suite is available. It wraps around library pynvml
    to extract relevant information.
    """

    def __init__(self):
        """Initialize the class."""
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

    def sample(self):
        """Extract gpu and memory usage.

        Args:
            [none]

        Returns:
              sample: a pandas series with gpu usage and ram measurements.
        """
        sample_series = pd.Series(dtype=float)
        sample_series["time"] = pd.Timestamp.now()

        memory_sample = psutil.virtual_memory()
        sample_series['ram (%)'] = memory_sample.percent
        for i in range(self.device_count):
            postfix = f"(GPU[{i}])"

            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            sample_series["gpu (%) " + postfix] = float(utilization.gpu)
            sample_series["mem (%) " + postfix] = float(utilization.memory)

        return sample_series

    def shutdown(self):
        """Shutdown the pynvml library."""
        pynvml.nvmlShutdown()


def append_to_file(file_path: Path, list_series: List[pd.Series]):
    """Append series to csv file.

    This function creates a csv file at file_path if it doesn't exist, or else
    it appends to the file if it exists.

    Args:
        file_path: path to the output csv file.
        list_series: list of pandas series to be outputed to csv file, row by row.
    """
    df = pd.DataFrame(list_series)

    if file_path.exists():
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)


def monitor(
    output_csv_file_path: Path,
    sleep_time_interval: float = 0.1,
    write_time_interval: float = 1.0,
    total_monitoring_interval: float = _eight_hours,
):
    """Monitor GPUs periodically.

    Args:
        output_csv_file_path: path of csv file where gpu measurements should be outputted.
        sleep_time_interval: time between gpu measurements, in seconds
        write_time_interval: time between output file flushes, in seconds
        total_monitoring_interval: total time during which there is monitoring, in seconds

    Side Effects:
        creates and periodically updates a csv file at output_csv_file_path.
    """
    gpu_monitor = GpuMonitor()

    global_start_time = time.time()
    global_time_difference = 0.0

    while global_time_difference < total_monitoring_interval:

        start_time = time.time()

        time_difference = 0.0

        list_sample_series = []
        while time_difference < write_time_interval:
            sample_series = gpu_monitor.sample()
            list_sample_series.append(sample_series)
            time.sleep(sleep_time_interval)
            time_difference = time.time() - start_time

        append_to_file(output_csv_file_path, list_sample_series)

        global_time_difference = time.time() - global_start_time

    gpu_monitor.shutdown()
