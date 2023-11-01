import abc
import typing

import segmentation_models_pytorch as smp
import torch.utils.data

import hardpicks.models.base as models_base
import hardpicks.models.constants as generic_consts
import hardpicks.models.losses as model_losses
import hardpicks.utils.hp_utils as hp_utils


class BaseSegmModel(models_base.BaseModel):
    """Base PyTorch-Lightning segmentation model interface.

    Compared to the vanilla base interface, this interface is specialized for image segmentation
    tasks and offers the automatic rendering/logging of prediction images each epoch.
    """

    def __init__(
        self,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any]
    ):
        """Validates+logs model hyperparameters and sets up the loss + metrics."""
        super().__init__(hyper_params)
        hp_utils.check_and_log_hp(
            names=[
                "head_class_count",
                "use_checkpointing",
                "segm_mask_field_name",
            ],
            hps=hyper_params,
        )
        self.use_checkpointing = hyper_params["use_checkpointing"]
        self.head_class_count = hyper_params["head_class_count"]
        self.segm_mask_field_name = hyper_params["segm_mask_field_name"]
        self.loss_fn = self._setup_loss_func(hyper_params)
        self.input_tensor_channels = self._get_expected_input_channel_count()

    @property
    def segm_class_count(self):
        """Returns the number of segmentation classes used in the classification head of the model."""
        # note: used only for backward compatibility with older scripts
        return self.head_class_count

    def _prepare_input_features(
        self,
        batch: typing.Any,
    ) -> torch.Tensor:
        """Returns the 'input feature tensor' to forward through the model for segmentation."""
        raise NotImplementedError  # must be defined in the derived class! (for fbp/simu/...)

    @abc.abstractmethod
    def _get_expected_input_channel_count(self):
        """Returns the expected channel count for the input tensors given the preproc config."""
        raise NotImplementedError  # must be defined in the derived class! (for fbp/simu/...)

    def _setup_loss_func(
        self,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ) -> typing.Callable:
        """Builds and returns the PyTorch-compatible loss function used in the training loop."""
        hp_utils.check_and_log_hp(
            names=[
                "loss_type",
            ],
            hps=hyper_params,
        )
        loss_type = hyper_params["loss_type"]
        loss_params = hyper_params.get("loss_params", {})
        # Set the loss mode to binary when there is a single class of interest: binary will mean
        # that we only care about the first class, and it's either that class or not. The loss
        # will ignore the other class.
        #
        # For 2+ classes, let's use multiclass: the loss will account for every class.
        if self.head_class_count == 1:
            loss_mode = smp.losses.BINARY_MODE
        else:
            loss_mode = smp.losses.MULTICLASS_MODE
        loss_ignore_index = generic_consts.DONTCARE_SEGM_MASK_LABEL
        return model_losses.get_loss_function(loss_type, loss_mode, loss_params, loss_ignore_index)
