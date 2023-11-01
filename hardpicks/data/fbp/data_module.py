import functools
import os
import typing

import numpy as np
import pytorch_lightning
import torch.utils.data
import torch.utils.data._utils

import hardpicks
from hardpicks.data.cache_utils import DatasetCacher
from hardpicks.data.fbp.collate import fbp_batch_collate
from hardpicks.data.fbp.consistent_dimensions_sampler import PowerTwoDimensionsGroupBatchSampler
from hardpicks.data.fbp.constants import SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP
from hardpicks.data.fbp.gather_wrappers import ShotLineGatherConcatDataset
from hardpicks.data.fbp.gather_cleaner import ShotLineGatherCleaner
from hardpicks.data.fbp.gather_parser import create_shot_line_gather_dataset
from hardpicks.data.fbp.gather_preprocess import ShotLineGatherPreprocessor
from hardpicks.data.fbp.gather_splitter import get_train_and_test_sub_datasets
from hardpicks.data.fbp.site_info import site_info_keys
from hardpicks.utils.hp_utils import check_and_log_hp


class FBPDataModule(pytorch_lightning.LightningDataModule):
    """Data Module for first break picking.

    This class will take in lists of dictionaries, one for each of train/valid/test. Each such
    dictionary is a "site_info" dictionary, which represents a site.
    """

    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and sets up internal attributes."""
        super().__init__()
        self.data_dir = data_dir

        site_info_array = hardpicks.data.fbp.site_info.get_site_info_array(data_dir)
        self.site_info_map = hardpicks.data.fbp.site_info.get_site_info_map(site_info_array)

        self.cache_dir = self._get_cache_dir(data_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        # first, check + load global hyperparams used for all data loaders
        check_and_log_hp([
            "train_batch_size",
            "eval_batch_size",
            "num_workers",
            "pin_memory",
            "convert_to_fp16",
            "convert_to_int16",
            "preload_trace_data",
            "cache_trace_metadata",
            "provide_offset_dists",
            "pad_to_nearest_pow2",
            "use_batch_sampler",
            "use_rebalancing_sampler",
            "segm_class_count",  # note: this is a model hyperparam needed here for preprocessing
            "skip_setup_overlap_check",
        ], hyper_params)
        self.train_batch_size = hyper_params["train_batch_size"]
        self.eval_batch_size = hyper_params["eval_batch_size"]
        self.num_workers = hyper_params["num_workers"]
        self.pin_memory = hyper_params["pin_memory"]
        self.dataset_hyper_params = dict(
            convert_to_fp16=hyper_params["convert_to_fp16"],
            convert_to_int16=hyper_params["convert_to_int16"],
            preload_trace_data=hyper_params["preload_trace_data"],
            cache_trace_metadata=hyper_params["cache_trace_metadata"],
            provide_offset_dists=hyper_params["provide_offset_dists"],
        )
        self.segm_class_count = hyper_params["segm_class_count"]
        assert self.segm_class_count in [None, *list(SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP.keys())], \
            "unexpected class count; should be None for regression or a supported value (int)"
        self.skip_setup_overlap_check = hyper_params["skip_setup_overlap_check"]
        self.collate_fn = functools.partial(
            fbp_batch_collate,
            pad_to_nearest_pow2=hyper_params["pad_to_nearest_pow2"],
        )
        self.use_batch_sampler = hyper_params["use_batch_sampler"]
        self.use_rebalancing_sampler = hyper_params["use_rebalancing_sampler"]  # only for training!
        assert not (self.use_batch_sampler and self.use_rebalancing_sampler), \
            "cannot use pow2 batch sampler and rebalancing sampler at the same time!"

        # next, check the train/valid/test loader parameter groups that will define which sites to use
        check_and_log_hp([
            "train_loader_params",
            "valid_loader_params",
            "test_loader_params",
        ], hyper_params)
        self._train_params = hyper_params["train_loader_params"]
        self._valid_params = hyper_params["valid_loader_params"]
        self._test_params = hyper_params["test_loader_params"]

        # note: the actual dataset instances are created in the 'setup' function below
        self._train_dataset, self._valid_dataset, self._test_dataset = None, None, None
        self._train_gather_names, self._valid_gather_names, self._test_gather_names = [], [], []

    def _get_cache_dir(self, data_dir):
        """Return the directory where the cache will be located."""
        return os.path.join(data_dir, "cache")

    def prepare_data(self):
        """Validates inputs.

        This method will be used to validate that the input site_info arrays are as expected and
        that the data it refers to exists.
        """
        for loader_params in [self._train_params, self._valid_params, self._test_params]:
            if not loader_params:
                continue
            assert isinstance(loader_params, list), "unexpected loader parameters type (need list)"
            for site_params in loader_params:
                assert "site_name" in site_params, "missing site name from loader params entry"
                site_name = site_params["site_name"]
                assert site_name in self.site_info_map, f"site '{site_name}' not found in known set"
                site_info = self.site_info_map[site_name]
                self._validate_site_info_dict(site_info)
                data_path = site_info["processed_hdf5_path"]
                assert os.path.exists(data_path), \
                    f"cannot find data for site '{site_name}' at: {data_path}"

    @staticmethod
    def _validate_site_info_dict(site_info: dict):
        for key in site_info_keys:
            assert key in site_info, f"{key} is missing from site_info dictionary"

    @staticmethod
    def create_parser(
        site_info: typing.Dict[typing.AnyStr, typing.Any],
        site_params: typing.Dict[typing.AnyStr, typing.Any],
        prefix: typing.AnyStr,
        dataset_hyper_params: typing.Dict[typing.AnyStr, typing.Any],
        segm_class_count: int,
    ):
        """Creates and returns a dataset parser (based on the `ShotLineGatherDataset` class)."""
        # note: the returned object will be wrapped in a bunch of other classes for preprocessing
        parser = create_shot_line_gather_dataset(
            hdf5_path=site_info["processed_hdf5_path"],
            site_name=site_info["site_name"],
            receiver_id_digit_count=site_info["receiver_id_digit_count"],
            first_break_field_name=site_info["first_break_field_name"],
            # plus the dataset hyperparameters: these should not change between sites
            **dataset_hyper_params,
        )
        auto_fill_missing_picks = site_params.get("auto_fill_missing_picks", None)
        assert not auto_fill_missing_picks or prefix not in ["valid", "test"], \
            "pick label auto-fill should *never* be activated in validation/testing!"
        parser = ShotLineGatherCleaner(
            parser,  # note: the 'cleaner' needs to always be the first wrapper applied to the parser
            # below are site-specific hyperparameters with None-defaults in case nothing is specified
            auto_invalidate_outlier_picks=site_params.get("auto_invalidate_outlier_picks", None),
            outlier_detection_strategy=site_params.get("outlier_detection_strategy", None),
            outlier_detection_filter_size=site_params.get("outlier_detection_filter_size", None),
            outlier_detection_threshold=site_params.get("outlier_detection_threshold", None),
            auto_fill_missing_picks=auto_fill_missing_picks,
            pick_fill_strategy=site_params.get("pick_fill_strategy", None),
            pick_fill_max_dist=site_params.get("pick_fill_max_dist", None),
            rejected_gather_yaml_path=site_params.get("rejected_gather_yaml_path", None),
        )
        default_generate_segm_masks = bool(segm_class_count)  # if not invalid/none, generate masks!
        generate_segm_masks = site_params.get("generate_segm_masks", default_generate_segm_masks)
        segm_first_break_buffer = site_params.get("segm_first_break_buffer", None)
        assert not segm_first_break_buffer or prefix not in ["valid", "test"], \
            "segmentation mask first break buffer should *never* be activated in validation/testing!"
        augmentations = site_params.get("augmentations", None)
        if isinstance(augmentations, dict):  # a dict may be used to allow indexing hyperparams w/ names
            augmentations = list(augmentations.values())  # we just discard the keys, keep the ops in order
        assert not augmentations or prefix not in ["valid", "test", "predict"], \
            "augmentations should *never* be activated in validation/testing! (not TTA impl yet)"
        parser = ShotLineGatherPreprocessor(
            parser,
            normalize_samples=site_params.get("normalize_samples", None),
            sample_norm_strategy=site_params.get("sample_norm_strategy", None),
            normalize_offsets=site_params.get("normalize_offsets", None),
            shot_to_rec_offset_norm_const=site_params.get("shot_to_rec_offset_norm_const", None),
            rec_to_rec_offset_norm_const=site_params.get("rec_to_rec_offset_norm_const", None),
            generate_first_break_prior_masks=site_params.get("generate_first_break_prior_masks", None),
            first_break_prior_velocity_range=site_params.get("first_break_prior_velocity_range", None),
            first_break_prior_offset_range=site_params.get("first_break_prior_offset_range", None),
            generate_segm_masks=generate_segm_masks,
            segm_class_count=segm_class_count,
            segm_first_break_buffer=site_params.get("segm_first_break_buffer", None),
            augmentations=augmentations,
        )
        if "subset" in site_params:
            # note: since we split after preproc, changes in preproc params might cause reshuffling!
            expected_subset_keys = ["eval_ratio", "use_eval_split"]
            assert all([key in site_params["subset"] for key in expected_subset_keys])
            # note: seed is not 'mandatory' here since a constant (0) will be OK for all sites
            split_rng = np.random.default_rng(seed=site_params["subset"].get("split_seed", 0))
            assert 0 < site_params["subset"]["eval_ratio"] < 1
            eval_ratio = site_params["subset"]["eval_ratio"]
            parser_train, parser_eval = get_train_and_test_sub_datasets(
                shot_line_gather_dataset=parser,
                random_number_generator=split_rng,
                fraction_of_shots_in_testing_set=eval_ratio,
                fraction_of_lines_in_testing_set=eval_ratio,
                ignore_line_ids_if_unique=site_params["subset"].get("ignore_line_ids", False),
            )
            parser = parser_eval if site_params["subset"]["use_eval_split"] else parser_train
        return parser

    def _create_dataset(
        self,
        params: typing.Dict[typing.AnyStr, typing.Any],
        prefix: typing.AnyStr,
    ):
        assert prefix in ["train", "valid", "test"]  # just used for asserts below
        if not params:
            return None
        list_site_dataset = []
        for site_params in params:
            site_name = site_params["site_name"]
            site_info = self.site_info_map[site_name]
            use_cache, cacher, parser = site_params.get("use_cache", True), None, None
            if use_cache:
                assert not self.dataset_hyper_params["preload_trace_data"], \
                    "cannot combine dataset caching + preloading (disk space would go boom!)"
                cache_hyper_params = {
                    "site_info": site_info,
                    "site_params": site_params,
                    "dataset_hyper_params": self.dataset_hyper_params,
                    "segm_class_count": self.segm_class_count,
                }
                cacher = DatasetCacher(
                    hyperparams=cache_hyper_params,
                    cache_dir_path=self.cache_dir,
                    cache_name_prefix=site_name,
                    # TODO: figure out if we want to keep scanning the hash every time...
                    # (it takes 60+seconds on an SSD, and could be replaced by file size check)
                    # on_disk_file_paths=[site_info["processed_hdf5_path"]],
                )
                if cacher.is_cache_available():
                    parser = cacher.load()
            if parser is None:
                parser = self.create_parser(
                    site_info=site_info,
                    site_params=site_params,
                    prefix=prefix,
                    dataset_hyper_params=self.dataset_hyper_params,
                    segm_class_count=self.segm_class_count,
                )
                if cacher is not None:
                    cacher.save(parser)
            list_site_dataset.append(parser)
        return ShotLineGatherConcatDataset(list_site_dataset)

    def setup(self, stage: typing.Optional[str] = None):
        """Sets up the data for the current stage, either fit or test."""
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self._train_dataset = self._create_dataset(self._train_params, "train")
            self._valid_dataset = self._create_dataset(self._valid_params, "valid")
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self._test_dataset = self._create_dataset(self._test_params, "test")
        if not self.skip_setup_overlap_check:
            # here, we do an exhaustive check that all gathers (identified by shot-recline) are not
            # leaked across train/validation/test datasets (no need to run this each session)
            if not self._train_gather_names and self._train_dataset is not None:
                self._train_gather_names = self._get_gather_names(self._train_dataset)
            if not self._valid_gather_names and self._valid_dataset is not None:
                self._valid_gather_names = self._get_gather_names(self._valid_dataset)
            if not self._test_gather_names and self._test_dataset is not None:
                self._test_gather_names = self._get_gather_names(self._test_dataset)
            assert len(np.intersect1d(self._train_gather_names, self._valid_gather_names)) == 0
            assert len(np.intersect1d(self._train_gather_names, self._test_gather_names)) == 0
            assert len(np.intersect1d(self._valid_gather_names, self._test_gather_names)) == 0

    def _get_gather_names(self, dataset) -> typing.List[typing.AnyStr]:
        """Returns the list of (hopefully!) unique gather names in a dataset."""
        site_names = list(self.site_info_map.keys())
        gather_names = []
        for gather_idx in range(len(dataset)):
            meta = dataset.get_meta_gather(gather_idx)
            try:
                origin_name = next(site_name for site_name in site_names if site_name in meta["origin"])
            except StopIteration:
                origin_name = meta["origin"]
            gather_id, shot_id, rec_line_id = meta["gather_id"], meta["shot_id"], meta["rec_line_id"]
            gather_names.append(f"{origin_name}_g{gather_id}_s{shot_id}_r{rec_line_id}")
        return gather_names

    def train_dataloader(self):
        """Creates the training dataloader."""
        if self._train_dataset is None:
            return None
        # todo: if we want training to be truly deterministic, we'll need to set seeds in worker inits
        dataloader_parameters = dict(
            dataset=self._train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            worker_init_fn=None
        )
        if self.use_batch_sampler:
            dataloader_parameters["shuffle"] = False
            dataloader_parameters["batch_sampler"] = PowerTwoDimensionsGroupBatchSampler(
                self._train_dataset,
                batch_size=self.train_batch_size,
            )
        elif self.use_rebalancing_sampler:
            assert isinstance(self._train_dataset, ShotLineGatherConcatDataset)
            sample_weights = self._train_dataset.get_sample_weights()
            dataloader_parameters["shuffle"] = False
            dataloader_parameters["batch_size"] = self.train_batch_size
            dataloader_parameters["sampler"] = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self._train_dataset),
                replacement=True,
            )
        else:
            dataloader_parameters["shuffle"] = True
            dataloader_parameters["batch_size"] = self.train_batch_size
        data_loader = torch.utils.data.DataLoader(**dataloader_parameters)
        return data_loader

    def val_dataloader(self):
        """Creates the validation dataloader."""
        if self._valid_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            dataset=self._valid_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            # todo: if we want training to be truly deterministic, we'll need to set seeds in worker inits
            worker_init_fn=None,
        )

    def test_dataloader(self):
        """Creates the test dataloader."""
        if self._test_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            dataset=self._test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            # todo: if we want training to be truly deterministic, we'll need to set seeds in worker inits
            worker_init_fn=None,
        )
