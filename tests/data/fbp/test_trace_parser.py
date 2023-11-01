import numpy as np

from hardpicks.data.fbp.trace_parser import RawTraceDataset


def test_raw_trace_dataset_get_first_break_indices():

    sample_rate_in_milliseconds = 2.0

    fbp_times_in_milliseconds = np.array([24.0, 25.0, 8.0, 1.0, 0.0, 2.0])
    expected_indices = np.array([12, 12, 4, 1, 0, 1])

    computed_indices = RawTraceDataset._get_first_break_indices(
        fbp_times_in_milliseconds, sample_rate_in_milliseconds
    )

    np.testing.assert_array_equal(computed_indices, expected_indices)
