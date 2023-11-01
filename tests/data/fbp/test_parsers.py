import logging
import os
import tempfile

import numpy as np
import pytest

import hardpicks.data.fbp.trace_parser
from tests.data.fbp.data_utils import create_fake_traces_and_hdf5_file


@pytest.fixture(scope="session")
def fake_dataset_params(first_break_field_name):
    return dict(
        samp_num=10,
        samp_rate=20,
        coord_scale=2,
        ht_scale=3,
        rec_count=30,
        rec_line_count=3,
        rec_id_digit_count=3,
        rec_peg_digit_count=6,
        shot_count=20,
        shot_id_digit_count=4,
        shot_peg_digit_count=7,
        first_break_field_name=first_break_field_name
    )


@pytest.fixture(scope="session")
def fake_dataset(fake_dataset_params):
    seed = 123
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_traces, output_hdf5_path = create_fake_traces_and_hdf5_file(fake_dataset_params, seed, tmp_dir)
        yield fake_traces, output_hdf5_path
    logging.info("Deleting test folder")


def _get_rec_line_id_from_peg(peg: int, digit_count: int) -> int:
    return peg // 10 ** digit_count


def _check_dataset_attribs(dataset, fake_dataset, params):
    assert len(dataset) == len(fake_dataset)
    assert len(dataset) <= params["rec_count"] * params["shot_count"]
    assert dataset.total_trace_count == len(fake_dataset)
    assert len(dataset.trace_to_shot_map) == len(fake_dataset)
    assert len(dataset.trace_to_line_map) == len(fake_dataset)
    assert len(dataset.trace_to_rec_map) == len(fake_dataset)
    assert len(dataset.trace_to_gather_map) == len(fake_dataset)
    assert dataset.receiver_id_digit_count == params["rec_id_digit_count"]
    assert dataset.samp_num == params["samp_num"]
    assert dataset.samp_rate == params["samp_rate"] * 1000
    assert len(dataset.first_break_labels) == len(fake_dataset)
    assert len(dataset.first_break_timestamps) == len(fake_dataset)
    assert dataset.coord_scale == params["coord_scale"]
    assert dataset.ht_scale == params["ht_scale"]


def _check_dataset_mappings(dataset, fake_dataset, params):
    expected_gather_count = params["shot_count"] * params["rec_line_count"]
    assert len(dataset.gather_to_trace_map) == expected_gather_count
    assert len(dataset.shot_to_trace_map) == params["shot_count"]

    # all trace ids should match 1:1 after parsing
    for trace_idx in range(len(dataset)):
        parsed_trace = dataset[trace_idx]
        orig_trace = fake_dataset[trace_idx]
        assert orig_trace["shot_peg"] == parsed_trace["shot_id"]
        assert orig_trace["rec_peg"] == parsed_trace["rec_id"]
        assert dataset.trace_to_shot_map[trace_idx] == parsed_trace["shot_id"]
        assert dataset.trace_to_line_map[trace_idx] == parsed_trace["rec_line_id"]
        assert dataset.trace_to_rec_map[trace_idx] == parsed_trace["rec_id"]
        assert dataset.trace_to_gather_map[trace_idx] == parsed_trace["gather_id"]

    # reverse shot mapping checks
    for shot_id, shot_trace_idxs in dataset.shot_to_trace_map.items():
        for trace_idx in shot_trace_idxs:
            assert dataset[trace_idx]["shot_id"] == shot_id
            assert fake_dataset[trace_idx]["shot_peg"] == shot_id
            assert dataset[trace_idx]["rec_id"] == fake_dataset[trace_idx]["rec_peg"]

    # reverse line mapping checks
    for line_id, line_trace_idxs in dataset.line_to_trace_map.items():
        for trace_idx in line_trace_idxs:
            assert dataset[trace_idx]["rec_line_id"] == line_id
            assert dataset[trace_idx]["shot_id"] == fake_dataset[trace_idx]["shot_peg"]
            assert dataset[trace_idx]["rec_id"] == fake_dataset[trace_idx]["rec_peg"]
            parsed_line_id = _get_rec_line_id_from_peg(
                dataset[trace_idx]["rec_id"], params["rec_id_digit_count"])
            assert dataset[trace_idx]["rec_line_id"] == parsed_line_id

    # reverse receiver mapping checks
    for rec_id, rec_trace_idxs in dataset.rec_to_trace_map.items():
        for trace_idx in rec_trace_idxs:
            assert dataset[trace_idx]["rec_id"] == rec_id
            assert fake_dataset[trace_idx]["rec_peg"] == rec_id
            assert dataset[trace_idx]["shot_id"] == fake_dataset[trace_idx]["shot_peg"]

    # reverse gather mapping checks
    for gather_id, gather_trace_idxs in dataset.gather_to_trace_map.items():
        for trace_idx in gather_trace_idxs:
            assert dataset[trace_idx]["gather_id"] == gather_id
            assert dataset[trace_idx]["rec_id"] == fake_dataset[trace_idx]["rec_peg"]
            assert dataset[trace_idx]["shot_id"] == fake_dataset[trace_idx]["shot_peg"]


@pytest.mark.parametrize("first_break_field_name", ['SPARE1', 'SPARE2', 'SPARE3', 'SPARE4'], scope='session')
def test_raw_trace_dataset(fake_dataset, fake_dataset_params, first_break_field_name):
    fake_dataset, fake_dataset_path = fake_dataset
    assert isinstance(fake_dataset, list) and len(fake_dataset)
    assert os.path.isfile(fake_dataset_path)
    parsed_dataset = hardpicks.data.fbp.trace_parser.create_raw_trace_dataset(
        hdf5_path=fake_dataset_path,
        receiver_id_digit_count=fake_dataset_params["rec_id_digit_count"],
        first_break_field_name=first_break_field_name,
        preload_trace_data=False,
    )

    _check_dataset_attribs(parsed_dataset, fake_dataset, fake_dataset_params)
    assert not parsed_dataset.preload_trace_data and not hasattr(parsed_dataset, "data_array")
    _check_dataset_mappings(parsed_dataset, fake_dataset, fake_dataset_params)

    preloaded_dataset = hardpicks.data.fbp.trace_parser.create_raw_trace_dataset(
        hdf5_path=fake_dataset_path,
        receiver_id_digit_count=fake_dataset_params["rec_id_digit_count"],
        first_break_field_name=first_break_field_name,
        preload_trace_data=True,
    )

    _check_dataset_attribs(preloaded_dataset, fake_dataset, fake_dataset_params)
    assert preloaded_dataset.preload_trace_data and hasattr(preloaded_dataset, "data_array")
    _check_dataset_mappings(preloaded_dataset, fake_dataset, fake_dataset_params)

    # for any trace, the overlap between the two datasets should be 100%
    trace_idx = np.random.randint(len(parsed_dataset))
    parsed_trace = parsed_dataset[trace_idx]
    preloaded_trace = preloaded_dataset[trace_idx]
    assert np.array_equal(parsed_trace.keys(), preloaded_trace.keys())
    perfect_overlap_keys = [k for k in parsed_trace.keys() if k != "samples"]
    for key in perfect_overlap_keys:
        assert np.array_equal(parsed_trace[key], preloaded_trace[key])
    assert np.allclose(parsed_trace["samples"].astype(np.float16), preloaded_trace["samples"])
