import logging
import tempfile
from typing import List

import h5py
import mock
import numpy as np
import pytest

from hardpicks.data.cache_utils import DatasetCacher
from hardpicks.data.fbp.constants import (
    BAD_FIRST_BREAK_PICK_INDEX,
)
from hardpicks.data.fbp.data_module import FBPDataModule
from hardpicks.utils.hash_utils import get_hash_from_path
from tests.data.fbp.data_utils import create_fake_traces_and_hdf5_file

fake_params1 = dict(
    samp_num=10,
    samp_rate=20,
    coord_scale=2,
    ht_scale=3,
    rec_count=30,
    rec_line_count=2,
    rec_id_digit_count=3,
    rec_peg_digit_count=6,
    shot_count=20,
    shot_id_digit_count=4,
    shot_peg_digit_count=7,
    first_break_field_name='SPARE1'
)

fake_params2 = dict(
    samp_num=12,
    samp_rate=11,
    coord_scale=3,
    ht_scale=4,
    rec_count=40,
    rec_line_count=4,
    rec_id_digit_count=4,
    rec_peg_digit_count=8,
    shot_count=10,
    shot_id_digit_count=3,
    shot_peg_digit_count=5,
    first_break_field_name='SPARE2'
)

fake_params3 = dict(
    samp_num=8,
    samp_rate=9,
    coord_scale=3,
    ht_scale=2,
    rec_count=50,
    rec_line_count=3,
    rec_id_digit_count=3,
    rec_peg_digit_count=6,
    shot_count=8,
    shot_id_digit_count=3,
    shot_peg_digit_count=5,
    first_break_field_name='SPARE3'
)

fake_params4 = dict(
    samp_num=24,
    samp_rate=10,
    coord_scale=3,
    ht_scale=5,
    rec_count=25,
    rec_line_count=1,
    rec_id_digit_count=3,
    rec_peg_digit_count=5,
    shot_count=18,
    shot_id_digit_count=3,
    shot_peg_digit_count=5,
    first_break_field_name='SPARE4'
)


def _get_list_site_info(list_seeds, list_site_names, list_fake_params, tmp_dir):
    list_site_info = []
    for seed, site_name, fake_params in zip(
        list_seeds, list_site_names, list_fake_params
    ):
        _, output_path = create_fake_traces_and_hdf5_file(fake_params, seed, tmp_dir)
        checksum = get_hash_from_path(output_path)
        site_info = dict(
            site_name=site_name,
            raw_hdf5_path=output_path,
            processed_hdf5_path=output_path,
            raw_md5_checksum=checksum,
            processed_md5_checksum=checksum,
            receiver_id_digit_count=fake_params["rec_id_digit_count"],
            first_break_field_name=fake_params["first_break_field_name"]
        )
        list_site_info.append(site_info)
    return list_site_info


def convert_site_info_arrays_to_config_dict(train_info, valid_info, test_info):
    # here, we'll convert the site info arrays into hyperparameter dictionaries
    hyper_params = {}
    new_site_info_map = {}
    for prefix, info in zip(
        ["train", "valid", "test"], [train_info, valid_info, test_info]
    ):
        site_list = []
        for site in info:
            site_dict = {
                "site_name": site["site_name"],
                "use_cache": True,
                "normalize_samples": False,
                "normalize_offsets": False,
            }  # other hyperparams don't matter
            site_list.append(site_dict)
            assert site["site_name"] not in new_site_info_map
            new_site_info_map[site["site_name"]] = {
                "site_name": site["site_name"],
                "raw_hdf5_path": site["raw_hdf5_path"],
                "processed_hdf5_path": site["processed_hdf5_path"],
                "receiver_id_digit_count": site["receiver_id_digit_count"],
                "first_break_field_name": site["first_break_field_name"],
                "raw_md5_checksum": site["raw_md5_checksum"],
                "processed_md5_checksum": site["processed_md5_checksum"],
            }
        hyper_params[prefix + "_loader_params"] = site_list if site_list else None
    return hyper_params, new_site_info_map


@pytest.mark.parametrize("batch_size", [1, 2], scope="class")
class TestFBPDataModule:
    @pytest.fixture(scope="class")
    def temporary_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logging.info("Creating temporary directory for test")
            yield tmp_dir
        logging.info("Deleting temporary directory")

    @pytest.fixture(scope="class")
    def shared_hyperparams_dict(self, batch_size):
        # these don't really matter for data loader tests, we'll just use static values
        return dict(
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            convert_to_fp16=True,
            convert_to_int16=True,
            preload_trace_data=False,
            cache_trace_metadata=False,
            provide_offset_dists=True,
            pad_to_nearest_pow2=False,
            use_batch_sampler=False,
            use_rebalancing_sampler=False,
            segm_class_count=None,
            skip_setup_overlap_check=False,
        )

    @pytest.fixture(scope="class")
    def training_site_info_array(self, temporary_directory):
        list_seeds = [123, 456]
        list_site_names = ["train1", "train2"]
        list_fake_params = [fake_params1, fake_params2]
        return _get_list_site_info(
            list_seeds, list_site_names, list_fake_params, temporary_directory
        )

    @pytest.fixture(scope="class")
    def validation_site_info_array(self, temporary_directory):
        list_seeds = [789]
        list_site_names = ["valid1"]
        list_fake_params = [fake_params3]
        return _get_list_site_info(
            list_seeds, list_site_names, list_fake_params, temporary_directory
        )

    @pytest.fixture(scope="class")
    def tst_site_info_array(self, temporary_directory):
        list_seeds = [1111]
        list_site_names = ["test1"]
        list_fake_params = [fake_params4]
        return _get_list_site_info(
            list_seeds, list_site_names, list_fake_params, temporary_directory
        )

    @pytest.fixture(scope="class")
    def corrupted_test_site_info_array(self):
        corrupted_site_info = dict(
            site_name="doesnotexist",
            raw_hdf5_path="/does/not/exist/",
            processed_hdf5_path="/does/not/exist/",
            raw_md5_checksum="aaaa",
            processed_md5_checksum="bbbb",
            receiver_id_digit_count=6,
            first_break_field_name='SPARE1',
        )
        return [corrupted_site_info]

    def test_corrupted_data_module_preparation(
        self,
        training_site_info_array,
        validation_site_info_array,
        corrupted_test_site_info_array,
        shared_hyperparams_dict,
        temporary_directory,
    ):

        hyper_params, new_site_info_map = convert_site_info_arrays_to_config_dict(
            training_site_info_array,
            validation_site_info_array,
            corrupted_test_site_info_array,
        )

        total_hyper_params = {**hyper_params, **shared_hyperparams_dict}
        with pytest.raises(AssertionError):
            _ = self._get_mocked_data_module(
                temporary_directory, total_hyper_params, new_site_info_map
            )

    def _get_data_module(
        self,
        shared_hyperparams_dict,
        temporary_directory,
        training_site_info_array,
        tst_site_info_array,
        validation_site_info_array,
    ):
        hyper_params, new_site_info_map = convert_site_info_arrays_to_config_dict(
            training_site_info_array, validation_site_info_array, tst_site_info_array
        )
        total_hyper_params = {**hyper_params, **shared_hyperparams_dict}
        data_module = self._get_mocked_data_module(
            temporary_directory, total_hyper_params, new_site_info_map
        )
        return data_module

    @staticmethod
    def _get_mocked_data_module(temporary_directory, hyper_params, new_site_info_map):
        # we need to replace the constant site maps to make this dummy test work

        # we need to replace the site maps to make this dummy test work
        patch_target1 = (
            "hardpicks.data.fbp.site_info.get_site_info_array"
        )
        patch_target2 = (
            "hardpicks.data.fbp.site_info.get_site_info_map"
        )
        # we also need to prevent mlflow from logging hyperparameters (otherwise it crashes)
        patch_target3 = (
            "hardpicks.data.fbp.data_module.check_and_log_hp"
        )

        with mock.patch(patch_target1), mock.patch(
            patch_target2, return_value=new_site_info_map
        ), mock.patch(patch_target3):
            data_module = FBPDataModule(
                data_dir=temporary_directory, hyper_params=hyper_params
            )

            data_module.prepare_data()
            data_module.setup()
        return data_module

    @pytest.fixture(scope="class")
    def data_module(
        self,
        training_site_info_array,
        validation_site_info_array,
        tst_site_info_array,
        shared_hyperparams_dict,
        temporary_directory,
    ):
        """This is the ACT fixture that creates the object to test against."""
        data_module = self._get_data_module(
            shared_hyperparams_dict,
            temporary_directory,
            training_site_info_array,
            tst_site_info_array,
            validation_site_info_array,
        )
        return data_module

    @staticmethod
    def _get_datasets_dictionaries(site_info_array: List[dict]) -> List:
        """Here we'll load the relevant small test data into RAM to avoid slow I/O against the hdf5 file."""
        list_dataset_dictionaries = []

        for site_info in site_info_array:
            data_file = h5py.File(site_info["processed_hdf5_path"], "r")
            first_break_field_name = site_info["first_break_field_name"]

            base_group = data_file["/TRACE_DATA/DEFAULT/"]
            dataset_dict = {
                "SHOT_PEG": base_group["SHOT_PEG"][:],
                "REC_PEG": base_group["REC_PEG"][:],
                "data_array": base_group["data_array"][:],
                "first_break_picks": base_group[first_break_field_name][:],
            }
            list_dataset_dictionaries.append(dataset_dict)
            data_file.close()

        return list_dataset_dictionaries

    @pytest.fixture(scope="class")
    def training_datasets_dictionaries(self, training_site_info_array):
        return self._get_datasets_dictionaries(training_site_info_array)

    @pytest.fixture(scope="class")
    def validation_datasets_dictionaries(self, validation_site_info_array):
        return self._get_datasets_dictionaries(validation_site_info_array)

    @pytest.fixture(scope="class")
    def tst_datasets_dictionaries(self, tst_site_info_array):
        return self._get_datasets_dictionaries(tst_site_info_array)

    def test_train_dataloader(self, data_module, training_datasets_dictionaries):
        data_loader = data_module.train_dataloader()
        self._assert_samples_are_in_dataset(data_loader, training_datasets_dictionaries)

    def test_validation_dataloader(self, data_module, validation_datasets_dictionaries):
        data_loader = data_module.val_dataloader()
        self._assert_samples_are_in_dataset(
            data_loader, validation_datasets_dictionaries
        )

    def test_test_dataloader(self, data_module, tst_datasets_dictionaries):
        data_loader = data_module.test_dataloader()
        self._assert_samples_are_in_dataset(data_loader, tst_datasets_dictionaries)

    def _assert_samples_are_in_dataset(self, data_loader, datasets_dictionaries):
        for batch in data_loader:
            shot_ids = batch["shot_id"].numpy()
            batch_recorder_ids = batch["rec_ids"].numpy()
            samples = batch["samples"].numpy()
            first_break_timestamps = batch["first_break_timestamps"].numpy()

            if "filled_first_breaks_mask" in batch:
                original_fbp_masks = ~batch["filled_first_breaks_mask"].numpy()
            else:
                original_fbp_masks = np.ones_like(first_break_timestamps).astype(bool)

            for shot_id, recorder_ids, computed_sample, computed_fbp, fbp_mask in zip(
                shot_ids,
                batch_recorder_ids,
                samples,
                first_break_timestamps,
                original_fbp_masks,
            ):
                expected_sample, expected_fbp = self._get_expected_sample_and_picks(
                    recorder_ids, shot_id, datasets_dictionaries
                )

                # If the batch size is larger than 1, the computed sample could be padded.
                number_of_batch_traces = computed_sample.shape[0]
                number_of_batch_time_samples = computed_sample.shape[1]

                number_of_real_traces = expected_sample.shape[0]
                number_of_real_time_samples = expected_sample.shape[1]

                assert expected_fbp.shape[0] == number_of_real_traces

                assert number_of_batch_traces >= number_of_real_traces
                assert number_of_batch_time_samples >= number_of_real_time_samples

                size_adjusted_computed_sample = computed_sample[
                    :number_of_real_traces, :number_of_real_time_samples
                ]
                size_adjusted_computed_fbp = computed_fbp[:number_of_real_traces]
                size_adjusted_fbp_mask = fbp_mask[:number_of_real_traces]

                # check that whatever is left over is padding
                if number_of_batch_traces > number_of_real_traces:
                    np.testing.assert_equal(
                        computed_fbp[number_of_real_traces:], BAD_FIRST_BREAK_PICK_INDEX
                    )

                if (
                    number_of_batch_traces > number_of_real_traces
                    or number_of_batch_time_samples > number_of_real_time_samples
                ):
                    padding_mask = np.ones(
                        [number_of_batch_traces, number_of_batch_time_samples]
                    ).astype(bool)
                    padding_mask[
                        :number_of_real_traces, :number_of_real_time_samples
                    ] = False
                    np.testing.assert_equal(computed_sample[padding_mask], 0.0)

                np.testing.assert_array_equal(
                    size_adjusted_computed_sample, expected_sample
                )
                np.testing.assert_array_equal(
                    size_adjusted_computed_fbp[size_adjusted_fbp_mask],
                    expected_fbp[size_adjusted_fbp_mask],
                )

    @staticmethod
    def _get_expected_sample_and_picks(recorder_ids, shot_id, datasets_dictionaries):
        correct_dataset_dict = None
        for dataset_dict in datasets_dictionaries:
            if shot_id in dataset_dict["SHOT_PEG"]:
                correct_dataset_dict = dataset_dict
        assert (
            correct_dataset_dict is not None
        ), "shot id does not belong to given datasets"
        shot_mask = np.isin(correct_dataset_dict["SHOT_PEG"], [shot_id])
        recorder_mask = np.isin(correct_dataset_dict["REC_PEG"], recorder_ids)
        gather_mask = shot_mask & recorder_mask
        expected_sample = correct_dataset_dict["data_array"][gather_mask]
        expected_first_break_picks = correct_dataset_dict["first_break_picks"][gather_mask]
        return expected_sample, expected_first_break_picks

    def test_cacher_in_data_module_is_called(
        self,
        data_module,
        training_site_info_array,
        validation_site_info_array,
        tst_site_info_array,
        shared_hyperparams_dict,
        temporary_directory,
    ):
        """This test confirms that, if fed the exact same inputs, the data module object uses the cacher."""

        expected_number_of_cache_loads = (
            len(training_site_info_array)
            + len(validation_site_info_array)
            + len(tst_site_info_array)
        )

        with mock.patch.object(DatasetCacher, "load") as mock_method:
            _ = self._get_data_module(
                shared_hyperparams_dict,
                temporary_directory,
                training_site_info_array,
                tst_site_info_array,
                validation_site_info_array,
            )

        assert mock_method.call_count == expected_number_of_cache_loads

    def test_cacher_in_data_module_not_called_on_corruped_checksum(
        self,
        batch_size,
        data_module,
        training_site_info_array,
        validation_site_info_array,
        tst_site_info_array,
        shared_hyperparams_dict,
        temporary_directory,
    ):
        """This test confirms that, if a checksum has changed, the data module object does not use the cacher."""
        expected_number_of_cache_loads = 0

        list_of_corrupted_arrays = [[], [], []]
        for i, correct_array in enumerate(
            [training_site_info_array, tst_site_info_array, validation_site_info_array]
        ):
            for site_info in correct_array:
                corrupted_site_info = dict(site_info)
                # add the batch size in the bad checksum to avoid collision between iterations
                # over the batch size parameter
                corrupted_site_info[
                    "processed_md5_checksum"
                ] = f"somebadhash_{batch_size}"
                list_of_corrupted_arrays[i].append(corrupted_site_info)

        with mock.patch.object(DatasetCacher, "load") as mock_method:
            _ = self._get_data_module(
                shared_hyperparams_dict,
                temporary_directory,
                list_of_corrupted_arrays[0],
                list_of_corrupted_arrays[1],
                list_of_corrupted_arrays[2],
            )

        assert mock_method.call_count == expected_number_of_cache_loads
