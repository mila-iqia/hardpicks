import os
import tempfile
from pathlib import Path

import pytest

from hardpicks.data.fbp.gather_parser import create_shot_line_gather_dataset
from hardpicks.data.fbp.site_info import get_site_info_by_name
from tests.fake_fbp_data_utilities import create_fake_hdf5_dataset, create_fake_site_data, IntegerRange, \
    FakeSiteSpecifications


def pytest_addoption(parser):
    parser.addoption(
        "--quick", action="store_true", default=False, help="skip slow tests"
    )
    parser.addoption(
        "--slow", action="store_true", default=False, help="only perform slow tests"
    )
    parser.addoption(
        "--very_slow", action="store_true", default=False, help="only perform very slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--quick"):
        # --quick given in cli: skip slow tests
        skip = pytest.mark.skip(reason="--quick option must be absent to run")
        for item in items:
            if "slow" in item.keywords or "very_slow" in item.keywords:
                item.add_marker(skip)
    elif config.getoption("--slow"):
        # --slow given in cli: only do the slow tests
        skip = pytest.mark.skip(reason="--slow option must be present to run")
        for item in items:
            if "slow" not in item.keywords or "very_slow" in item.keywords:
                item.add_marker(skip)
    elif config.getoption("--very_slow"):
        # --very_slow given in cli: only do the very slow tests
        skip = pytest.mark.skip(reason="--very_slow option must be present to run")
        for item in items:
            if "very_slow" not in item.keywords:
                item.add_marker(skip)


@pytest.fixture()
def real_site_info(real_site_name):
    return get_site_info_by_name(real_site_name)


@pytest.fixture()
def real_site_path(real_site_info):
    return real_site_info["raw_hdf5_path"]


@pytest.fixture()
def real_dataset(real_site_info, real_site_path):
    if not os.path.isfile(real_site_path):
        return None  # it can't be loaded, the actual tests will have to skip
    dataset = create_shot_line_gather_dataset(
        hdf5_path=real_site_path,
        site_name=real_site_info["site_name"],
        receiver_id_digit_count=real_site_info["receiver_id_digit_count"],
        first_break_field_name=real_site_info["first_break_field_name"],
        preload_trace_data=False,
        provide_offset_dists=True,
    )
    return dataset


@pytest.fixture()
def fake_specs(receiver_id_digit_count):
    fake_site_specifications = FakeSiteSpecifications(
        site_name="test",
        random_seed=42,
        samp_rate=4000,
        receiver_id_digit_count=receiver_id_digit_count,
        first_break_field_name="SPARE1",
        number_of_time_samples=100,
        number_of_shots=3,
        number_of_lines=10,
        shot_id_is_corrupted=False,
        range_of_receivers_per_line=IntegerRange(min=10, max=30),
    )
    return fake_site_specifications


@pytest.fixture()
def fake_data(fake_specs):
    return create_fake_site_data(fake_specs)


@pytest.fixture()
def hdf5_path(fake_specs):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        output_hdf5_path = Path(tmp_dir_str).joinpath("test.hdf5")
        fake_data = create_fake_site_data(fake_specs)
        create_fake_hdf5_dataset(fake_data, output_hdf5_path)
        yield output_hdf5_path
