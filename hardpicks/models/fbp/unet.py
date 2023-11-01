import copy
import typing

import cv2 as cv
import torch.utils.data

import hardpicks.data.fbp.constants as constants
import hardpicks.models.unet_base as unet_base
import hardpicks.models.fbp.utils as model_utils
import hardpicks.utils.hp_utils as hp_utils


class FBPUNet(unet_base.UNet):
    """UNet model derived specifically for first-break-picking applications.

    The role of this class is to offer backward compatibility between the base UNet class
    (`hardpicks.models.unet_base.UNet`) and the older config files
    that were previously used to run experiments.

    The current setup of this model only allows first-break-picking to be tackled as a
    classification (gather segmentation) problem. The constructor also receives extra FBP-
    related hyperparameters to control what data is assembled into an input tensor for the model.
    """

    @staticmethod
    def _update_hyper_params_for_backward_compatibility(
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Backward compatibility bandaid function used to bridge new and old config files for FBP."""
        hyper_params = copy.deepcopy(hyper_params)  # keeps everything intact outside the model
        if "gathers_to_display" in hyper_params and "images_to_display" not in hyper_params:
            hyper_params["images_to_display"] = hyper_params["gathers_to_display"]
        elif "images_to_display" in hyper_params:
            assert hyper_params["images_to_display"] == hyper_params["gathers_to_display"]
        if "head_class_count" not in hyper_params:
            assert "segm_class_count" in hyper_params  # this is what we need to re-map for the model
            segm_class_count = hyper_params["segm_class_count"]
            assert segm_class_count in constants.SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP
            hyper_params["head_class_count"] = \
                len(constants.SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP[segm_class_count])
        if "segm_mask_field_name" in hyper_params:
            assert hyper_params["segm_mask_field_name"] == "segmentation_mask", \
                "FBP model will auto-specify mask name"
        else:
            hyper_params["segm_mask_field_name"] = "segmentation_mask"
        return hyper_params

    def __init__(
        self,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates+logs model hyperparameters and sets up the loss + metrics."""
        hyper_params = self._update_hyper_params_for_backward_compatibility(hyper_params)
        hp_utils.check_and_log_hp(
            names=[
                "use_dist_offsets",
                "use_first_break_prior",
                "segm_first_break_prob_threshold",
            ],
            hps=hyper_params,
        )
        self.use_dist_offsets = hyper_params["use_dist_offsets"]
        self.use_first_break_prior = hyper_params["use_first_break_prior"]
        self.segm_first_break_prob_threshold = hyper_params["segm_first_break_prob_threshold"]
        assert self.segm_first_break_prob_threshold >= 0.0
        super().__init__(hyper_params)
        assert self.segm_class_count in constants.SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP

    def _prepare_input_features(
        self,
        batch: typing.Any,
    ) -> torch.Tensor:
        """Returns the 'input feature tensor' to forward through the model for FBP tasks."""
        input_tensor = model_utils.prepare_input_features(
            batch,
            self.use_dist_offsets,
            self.use_first_break_prior,
        )
        assert input_tensor.shape[1] == self.input_tensor_channels, "woopsie, messed up feature prep?"
        return input_tensor

    def _get_expected_input_channel_count(self):
        """Returns the expected channel count for the input tensors given the preproc config."""
        input_ch_count = 1  # 'samples' == 1 channel (by default)
        if self.use_dist_offsets:
            input_ch_count += 3
        if self.use_first_break_prior:
            input_ch_count += 1
        return input_ch_count

    def _get_persistent_data_id(
        self,
        batch: typing.Dict,
        batch_idx: int,  # index of the batch itself inside the epoch
        data_sample_idx: int,  # index of the data sample inside the batch we want to log
    ) -> typing.Any:
        """Returns the full 'identifier' (or name) used to uniquely tag a gather."""
        origin_name = batch["origin"][data_sample_idx]
        gather_id = batch["gather_id"][data_sample_idx].item()
        shot_id = batch["shot_id"][data_sample_idx].item()
        rec_line_id = batch["rec_line_id"][data_sample_idx].item()
        return f"{origin_name}_g{gather_id}_s{shot_id}_r{rec_line_id}"

    def _render_and_log_data_sample(
        self,
        batch: typing.Dict,
        batch_idx: int,  # index of the batch itself inside the epoch
        data_sample_idx: int,  # index of the data sample inside the batch we want to log
        data_id: typing.Any,  # the hashable identifier used to name this particular sample
        raw_preds: typing.Any,
        prefix: typing.AnyStr,
    ) -> None:
        """Generates and logs prediction/target images via the hidden tensorboard writer attribute."""
        image = model_utils.generate_pred_image(
            batch=batch,
            raw_preds=raw_preds,
            batch_gather_idx=data_sample_idx,
            segm_class_count=self.segm_class_count,
            segm_first_break_prob_threshold=self.segm_first_break_prob_threshold,
        )
        assert hasattr(self, "_tbx_logger"), "where did the tbx logging cheat attrib go? (check main)"
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # for tensorboard compatibility
        self._tbx_logger.experiment.add_image(
            f"{prefix}/{data_id}",
            image,
            self.global_step,
            dataformats="HWC",
        )
