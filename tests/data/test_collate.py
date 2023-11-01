import functools
import math

import mock
import numpy as np
import pytest
import torch.utils.data

import hardpicks.data.fbp.collate


@pytest.fixture(scope="session")
def fake_fields_to_double_pad():
    return ["var_double_array_field_1", "var_double_array_field_2"]


@pytest.fixture(scope="session")
def fake_fields_to_pad(fake_fields_to_double_pad):
    return [
        "var_array_field_1",
        "var_array_field_2",
        "var_array_field_3",
        "var_array_field_missing",
    ] + fake_fields_to_double_pad


@pytest.fixture(scope="session")
def fake_dataset(fake_fields_to_pad, fake_fields_to_double_pad):
    np.random.seed(123)
    # we'll just make a list of static samples to merge together
    dataset = []
    for _ in range(1024):
        sample_data = {
            "not_var_attrib1": "hello",
            "not_var_attrib2": {1: 0},
            "not_var_attrib3": 3.0,
            "not_var_attrib4": (1, 2, 3),
            "not_var_array1": np.random.randn(32),
            "not_var_array2": np.random.randn(1, 2, 3, 4),
        }
        var_array_dim1 = 1 + int(np.random.rand() * 100)
        for field_name in fake_fields_to_pad:
            if "missing" not in field_name:
                sample_data[field_name] = np.random.rand(var_array_dim1, 42)

        var_array_dim2 = 1 + int(np.random.rand() * 50)
        for field_name in fake_fields_to_double_pad:
            if "missing" not in field_name:
                sample_data[field_name] = np.random.rand(var_array_dim1, var_array_dim2)

        dataset.append(sample_data)
    return dataset


@pytest.mark.parametrize("pad_to_nearest_pow2", [True, False])
def test_fbp_collate(
    fake_dataset, fake_fields_to_pad, fake_fields_to_double_pad, pad_to_nearest_pow2
):
    # check whether the new collate does its job with variable-length arrays
    batch_size = 32
    collate_fn = hardpicks.data.fbp.collate.fbp_batch_collate
    collate_fn = functools.partial(collate_fn, pad_to_nearest_pow2=pad_to_nearest_pow2)
    dataloader = torch.utils.data.DataLoader(
        dataset=fake_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    mod_to_mock = hardpicks.data.fbp.collate
    map_for_mock1 = [(name, 0) for name in fake_fields_to_pad]
    map_for_mock2 = fake_fields_to_double_pad

    with mock.patch.object(mod_to_mock, "get_fields_to_pad", return_value=map_for_mock1), \
            mock.patch.object(mod_to_mock, "get_fields_to_double_pad", return_value=map_for_mock2):
        sample_idx_offset = 0
        for batch_idx, batch in enumerate(dataloader):
            expected_max_size = max(
                [
                    len(fake_dataset[sample_idx_offset + idx]["var_array_field_1"])
                    for idx in range(batch_size)
                ]
            )

            if pad_to_nearest_pow2:
                expected_max_size = int(
                    2 ** (math.ceil(math.log(expected_max_size, 2)))
                )
            for field_name in fake_fields_to_pad:
                if "missing" not in field_name:
                    assert len(batch[field_name]) == batch_size
                    assert batch[field_name].shape[1] == expected_max_size
            sample_idx_offset += batch_size
