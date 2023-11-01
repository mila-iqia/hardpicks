"""This module contains an encoder wrapper around the classic PyTorch ResNet implementation.

The goal of this wrapper is to allow for the instantiation of ResNet encoders with a bit more
control over the configuration of its layers (width, depth, ...). Obviously, when using this
wrapper, pretrained models from SMP, TIMM, or torchvision will no longer be available, unless
the layer/block/stage configurations match perfectly.
"""
import collections
import functools
import typing

import pretrainedmodels.models.senet  # this is an SMP-related dependency
import segmentation_models_pytorch.encoders._base as smp_encoders_base
import timm.models.resnest
import torch
import torchvision.models.resnet as torchvis_base

SUPPORTED_BLOCK_TYPES = {
    "BasicBlock": torchvis_base.BasicBlock,
    "Bottleneck": torchvis_base.Bottleneck,
    "SEBottleneck": pretrainedmodels.models.senet.SEBottleneck,
    "ResNestBottleneck": timm.models.resnest.ResNestBottleneck,
}

SUPPORTED_NORM_TYPES = {
    "BatchNorm": torch.nn.BatchNorm2d,
    "InstanceNorm": torch.nn.InstanceNorm2d,
    "LayerNorm": torch.nn.LayerNorm,
    "GroupNorm": torch.nn.GroupNorm,
}


class ResNetEncoder(smp_encoders_base.EncoderMixin, torch.nn.Module):
    """SMP-based ResNet encoder class.

    This class is inspired by SMP as well as the original torchvision ResNet implementation; see
    `segmentation_models_pytorch/encoders/resnet.py` and `torchvision/models/resnet.py` for more
    information. It supports TIMM-based block definitions based on the `SUPPORTED_BLOCK_TYPES`
    attribute.

    This version differs in how it can be used to customize the architecture more profoundly.
    Classic ResNet configurations can be provided as such:

        ResNet18:
            block: "BasicBlock"
            layers: [2, 2, 2, 2]
            channels: [64, 128, 256, 512]
            groups: 1
            width_per_group: 64
            norm_layer: "BatchNorm"

        ResNet34:
            block: "BasicBlock"
            layers: [3, 4, 6, 3]
            channels: [64, 128, 256, 512]
            groups: 1
            width_per_group: 64
            norm_layer: "BatchNorm"

        ResNet50:
            block: "Bottleneck"
            layers: [3, 4, 6, 3]
            channels: [64, 128, 256, 512]
            groups: 1
            width_per_group: 64
            norm_layer: "BatchNorm"

        WideResNet50:
            block: "Bottleneck"
            layers: [3, 4, 6, 3]
            channels: [64, 128, 256, 512]
            groups: 1
            width_per_group: 128
            norm_layer: "BatchNorm"

        # etc.

    Note that the 'depth' of the ResNet (in terms of number of stages) is always defined as the
    length of the `layers` or `channels` lists (which should be identical) plus one extra stage for
    the first convolution layer and another extra stage for the 'identity' block (used to copy over
    the input for final decoding e.g. in U-Nets).

    Finally, note that some attributes of this class (namely: `_out_channels`, `_depth`, `_in_channels`,
    and the `get_stages` function) are required for this encoder to be used in conjunction with
    decoders from the SMP library.
    """

    def __init__(
        self,
        block: typing.Union[typing.AnyStr, typing.Type],
        layers: typing.List[int],
        channels: typing.List[int],
        in_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: typing.Optional[typing.List[bool]] = None,
        norm_layer: typing.Optional[typing.Callable[..., torch.nn.Module]] = None,
        norm_layer_params: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
        block_params: typing.Optional[typing.Dict[typing.AnyStr, typing.Any]] = None,
    ) -> None:
        """Initializes the ResNet model (always from scratch).

        Args;
            block: block type to use in each layer across all stages.
            layers: number of layers to use in each stage, where each entry in the list corresponds to a stage.
            channels: number of channels to use when creating the last conv block of each stage.
            zero_init_residual: toggles whether to zero-init the norm layer in each residual branch.
            groups: number of blocked connections from input channels to output channels in conv layers.
            width_per_group: controls the width of channel groups when using bottleneck-based blocks.
            replace_stride_with_dilation: defines whether to replace the stride by a dilated convolution
                in each of the stages following the first two mandatory stages.
            norm_layer: type of normalization layer to use inside the encoder (including its blocks).
            norm_layer_params: dictionary of extra arguments passed to the constructor of the norm layers.
            block_params: dictionary of extra arguments passed to the constructor of the block layers.
        """
        super().__init__()

        assert len(layers) == len(channels) and len(layers) > 0, \
            "length of layer count and channel count lists should be identical"
        assert len(layers) >= 1, "should have at least two stages to build!"
        assert all([n > 0 for n in layers]) and all([n > 0 for n in channels]), \
            "all layer and channel counts should be strictly positive integers"
        self._depth = len(layers) + 1  # the first stage is assembled manually w/o layer/ch counts
        self._layers = layers
        self._in_channels = in_channels  # required by SMP for the internal setter functions
        self._channels = channels

        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        elif isinstance(norm_layer, str):
            norm_layer = SUPPORTED_NORM_TYPES[norm_layer]
        assert callable(norm_layer)
        if norm_layer_params is None:
            norm_layer_params = {}
        assert isinstance(norm_layer_params, dict)
        norm_layer_constr = functools.partial(norm_layer, **norm_layer_params)
        self._norm_layer = norm_layer_constr

        if block is None:
            block = torchvis_base.BasicBlock
        elif isinstance(block, str):
            block = SUPPORTED_BLOCK_TYPES[block]
        assert callable(block)
        assert hasattr(block, "expansion") and isinstance(block.expansion, int), \
            "missing mandatory 'expansion' class member for given block type"
        self._block_expansion = block.expansion
        if block_params is None:
            block_params = {}
        assert isinstance(block_params, dict)
        block_constr = functools.partial(block, **block_params)
        self._block = block_constr

        self.groups = groups
        self.base_width = width_per_group
        self._out_channels, self.inplanes, self.dilation = None, None, None  # updated in build call
        self._build(replace_stride_with_dilation=replace_stride_with_dilation)
        self._init_weights(zero_init_residual=zero_init_residual)

    def _build(
        self,
        replace_stride_with_dilation: typing.Optional[typing.List[bool]]
    ) -> None:
        """Builds the actual list of stages (which are torch.nn.Module-derived objects) for this model."""
        self.inplanes = self._channels[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element indicates if we should replace the 2x2 stride w/ a dilated convolution
            replace_stride_with_dilation = [False] * (len(self._layers) - 1)
        assert len(replace_stride_with_dilation) == len(self._layers) - 1, \
            "replace_stride_with_dilation should be a list for the N-1 last stages of the model"
        self._out_channels = [self._in_channels]  # for the ch count of the identity stage in forward
        # FIRST (MANDATORY) STAGE IS A SIMPLE CONV-BN-RELU BLOCK
        # (this stage is assembled into a torch.nn.Sequential module on-demand)
        self.conv1 = torch.nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self._out_channels.append(self.inplanes)
        # SECOND (MANDATORY) STAGE IS A MAXPOOL LAYER COMBINED WITH THE FIRST RESIDUAL LAYER MODULE
        # (this stage is assembled into a torch.nn.Sequential module on-demand)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self._block, self._channels[0], self._layers[0])
        self._out_channels.append(self._channels[0] * self._block_expansion)
        self.layers = []  # this is MEANT to not be a torch.nn.ModuleList(), as we use setattr below
        # SUBSEQUENT STAGES ARE OPTIONAL, AND DERIVED FROM THE LAYERS/CHANNELS LISTS
        for stage_offset_idx in range(1, len(self._layers)):
            curr_layer = self._make_layer(
                block=self._block,
                planes=self._channels[stage_offset_idx],
                blocks=self._layers[stage_offset_idx],
                stride=2,
                dilate=replace_stride_with_dilation[stage_offset_idx - 1],
            )
            # for backward compatibility, and to share attr names with other resnet impls
            setattr(self, f"layer{stage_offset_idx + 1}", curr_layer)
            self.layers.append(curr_layer)
            self._out_channels.append(self._channels[stage_offset_idx] * self._block_expansion)
        self._out_channels = tuple(self._out_channels)  # as required by SMP

    def _make_layer(
        self,
        block: typing.Callable,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> torch.nn.Sequential:
        """Builds a single layer based on the provided arguments (same as in the original impl)."""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:  # if using dilation instead of stride, adjust the two factors here
            self.dilation *= stride
            stride = 1
        # if using a stride or modifying the ch count, we need to 'scale' the skip connection too
        if stride != 1 or self.inplanes != planes * self._block_expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * self._block_expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * self._block_expansion),
            )
        layers = [block(  # first block in the layer is where input downsampling happens (if any)
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            groups=self.groups,
            base_width=self.base_width,
            dilation=previous_dilation,
            norm_layer=norm_layer,
        )]
        self.inplanes = planes * self._block_expansion
        for _ in range(1, blocks):  # other blocks are optional, depending on input layer count
            layers.append(block(
                inplanes=self.inplanes,
                planes=planes,
                # stride & downsample unspecified = kept as default values (there should be none)
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            ))
        return torch.nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool) -> None:
        """Initializes the weights for the internal submodules based on the requested strategy(ies)."""
        # first, use the kaiming-style initialization for all conv/norm layers
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch (same as orig resnet impl),
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                supported_bottleneck_types = (
                    timm.models.resnest.ResNestBottleneck,
                    pretrainedmodels.models.senet.SEBottleneck,
                    torchvis_base.Bottleneck,
                )
                if isinstance(m, supported_bottleneck_types):
                    assert hasattr(m, "bn3") and isinstance(m.bn3, torch.nn.BatchNorm2d)
                    torch.nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, torchvis_base.BasicBlock):
                    assert hasattr(m, "bn2") and isinstance(m.bn2, torch.nn.BatchNorm2d)
                    torch.nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def get_stages(self) -> typing.Sequence[torch.nn.Module]:
        """Returns the different processing stages of this encoder, as expected by SMP.

        Each stage should be a `torch.nn.Module`-compatible object. There will always be one more
        stage here than the actual depth of the network in order to provide an identity stage as the
        first entry (in case the decoder needs a copy of the original input).
        """
        return [
            torch.nn.Identity(),  # as required by SMP, in order to make the decoding easier for U-Nets
            torch.nn.Sequential(self.conv1, self.bn1, self.relu),  # mandatory stage, without ch expansion
            torch.nn.Sequential(self.maxpool, self.layer1),  # mandatory stage, with ch expansion
            *self.layers,  # subsequent stages, with ch expansion (typical resnet adds three stages here)
        ]

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        """Ingests an input tensor, passing it through all encoder stages, and returns the result."""
        features = []
        for stage in self.get_stages():
            x = stage(x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict: typing.OrderedDict[str, torch.Tensor], strict: bool = True):
        """Loads a state dictionary of parameters into the current model.

        Note: this will silently handle extra 'fc' layer parameters if those are specified in
        order for this encoder to be 100% compatible with torchvision-based ResNet checkpoints.
        """
        state_dict = collections.OrderedDict(
            # we do not have any fully-connected layer definition in this encoder!
            **{k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        )
        super().load_state_dict(state_dict=state_dict, strict=strict)
