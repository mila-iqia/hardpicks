import abc
import copy
import logging
import math
import typing

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base as smp_base
import torch
import torch.nn
import torch.nn.functional
import torch.utils.checkpoint
import torch.utils.data
try:
    import fairscale.nn
except ImportError:
    fairscale = None

import hardpicks.metrics.base as metrics_base
import hardpicks.models.coordconv as coordconv_utils
import hardpicks.models.resnet as custom_resnet
import hardpicks.models.segm_base as model_base
import hardpicks.utils.hp_utils as hp_utils
from hardpicks.utils.profiling_utils import profile

logger = logging.getLogger(__name__)

SUPPORTED_ENCODER_TYPES = [
    "vanilla",  # this is a simple CNN based on stacks of Conv2d-BN-ReLU layers only
    "CustomResNet", "custom-resnet", "resnet",  # all of these redirect to our custom resnet encoder
    *smp.encoders.encoders,
]

SUPPORTED_DECODER_TYPES = [
    "vanilla",  # this is a simple CNN based on stacks of Conv2d-BN-ReLU layers only
    # TODO: add more decoder styles here!
]


class Basic2DBlock(torch.nn.Module):
    """Base class for the 2D blocks.

    The forward method will be made abstract and must be defined in the concrete classes.
    The reason why we create an abstract base class is that the encoder block "forward" and
    the decoder block "forward" require different numbers of argument (the encoder takes one
    argument and returns many things, while the decoder block takes in two things (the upsampled
    and the skip).

    The simple workaround of passing a single argument that can either be a tensor or a tuple of
    tensor collides with fairscale's checkpoint_wrapper. The latter is smart enough to
    handle multiple inputs and outputs, but not smart enough to realize that we've wrapped our
    multiple inputs into a single tuple. To prevent further headaches, it is best to have explicit
    arguments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        coordconv: bool = False,
    ):
        """Constructs the double-layer block."""
        super().__init__()
        # note: we keep these args as attributes for debugging/easy access from other modules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coordconv = coordconv
        self.kernel_size = kernel_size
        self.padding = padding
        # below are the actual layer definitions for this block
        self.layer1 = torch.nn.Sequential(
            coordconv_utils.make_conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # no need since we have a batch norm layer right after
                coordconv=coordconv,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.layer2 = torch.nn.Sequential(
            coordconv_utils.make_conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # no need since we have a batch norm layer right after
                coordconv=coordconv,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Must be implemented in concrete class."""
        pass


class Basic2DEncoderBlock(Basic2DBlock):
    """Default encoder block."""

    def forward(self, x):
        """Forwards the tensor through all layers of this block."""
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Basic2DDecoderBlock(Basic2DBlock):
    """Default decoder block used in U-Net layers with learnable upsampling."""

    def __init__(
        self,
        prev_channels: int,
        upsampl_channels: int,  # typically half of the previous featmap channels
        skip_channels: int,  # typically equal to the upsampled featmap channels
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        coordconv: bool = False,
        attention: typing.Optional[typing.AnyStr] = None,
    ):
        """Constructs a decoder block on top of a standard double-layer block."""
        concat_channels = skip_channels + upsampl_channels
        super().__init__(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            coordconv=coordconv,
        )
        # note: we keep these args as attributes for debugging/easy access from other modules
        self.prev_channels = prev_channels
        self.upsampl_channels = upsampl_channels
        self.skip_channels = skip_channels
        # below are the actual layer definitions for this decoding block (+ the base)
        self.upsampler = torch.nn.ConvTranspose2d(  # this layer will 'learn' to do a 2x upsampling
            in_channels=prev_channels,
            out_channels=upsampl_channels,
            kernel_size=(2, 2),
            stride=(2, 2),
        )  # TODO: we could initialize the weights of this upsampling layer w/ gaussian kernel!
        self.att1 = smp_base.modules.Attention(  # will be identity fn if attention is not needed
            name=attention,
            in_channels=concat_channels,
        )
        self.att2 = smp_base.modules.Attention(  # will be identity fn if attention is not needed
            name=attention,
            in_channels=out_channels,
        )

    def forward(self, prev: torch.Tensor, skip: typing.Union[None, torch.Tensor]):
        """Decodes the combination of the previous layer and encoder feature maps."""
        assert prev.shape[1] == self.prev_channels
        x = self.upsampler(prev)
        if skip is None:
            assert self.skip_channels == 0
        else:
            assert skip.shape[1] == self.skip_channels
            x = torch.cat([skip, x], dim=1)
        x = self.att1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.att2(x)
        return x


class Basic2DEncoder(torch.nn.Module):
    """Default (vanilla) encoder that contains a basic blocks and generates a pyramid of feat maps."""

    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels: typing.Sequence[int] = tuple([16, 32, 64, 128, 256]),  # = vanilla unet
        coordconv: bool = False,
    ):
        """Constructs a series of double-layer blocks with increasing feature depth."""
        super().__init__()
        # note: we keep these args as attributes for debugging/easy access from other modules
        self.in_channels = in_channels
        assert len(block_out_channels) > 0, "encoder should have at least one block!"
        assert all([ch > 0 for ch in block_out_channels]), "all ch counts should be specified!"
        out_channels = [in_channels]  # this encoder will always return the input as a feature map
        blocks = []
        for block_out_ch in block_out_channels:
            assert block_out_ch >= out_channels[-1], "encoder that does not encode?"
            blocks.append(Basic2DEncoderBlock(
                in_channels=out_channels[-1],
                out_channels=block_out_ch,
                coordconv=coordconv,
            ))
            out_channels.append(block_out_ch)
        # note: we also return the downsampled map at the end with the same ch count as the last block
        out_channels.append(block_out_channels[-1])
        self.blocks = torch.nn.ModuleList(blocks)
        self.out_channels = out_channels

    def forward(self, x):
        """Encodes a tensor and returns its stack of (still-attached) feat maps (one per block)."""
        feat_maps = [x]  # the input is also a 'feature map' that could be useful for decoding
        for block in self.blocks:
            x = block(x)
            feat_maps.append(x)
            x = torch.nn.functional.max_pool2d(x, 2)
        feat_maps.append(x)
        return feat_maps  # we'll need an extra decoder block ('mid') to deal with the last one

    @property
    def block_count(self) -> int:
        """Returns the number of encoder blocks in this module."""
        return len(self.blocks)


class Basic2DDecoder(torch.nn.Module):
    """Default decoder that contains multiple blocks (upsamplers+layers) and a task-specific head."""

    def __init__(
        self,
        in_channels: typing.Sequence[int],  # needed to know what's incoming from the encoder
        mid_block_channels: int = 0,  # 0 == do not use a mid block (skip it, go straight decoder)
        block_out_channels: typing.Optional[typing.Sequence[int]] = None,  # None/empty = auto-fill
        head_class_count: typing.Optional[int] = None,  # None == do not use a classification head
        coordconv: bool = False,
        attention: typing.Optional[typing.AnyStr] = None,
        use_checkpointing: bool = False,
        use_skip_connections: bool = True,  # useful for debugging & for usage in auto encoders...
        # TODO: forward some conv2d-level args from here?
    ):
        """Constructs a series of decoder blocks with decreasing feature depth."""
        super().__init__()
        assert len(in_channels) > 1 and all([ch > 0 for ch in in_channels]), \
            "invalid channel counts for the input feature maps! (need input tensor + more maps!)"
        if mid_block_channels is not None and mid_block_channels > 0:
            assert len(in_channels) > 2, "also sending last featmap to mid block, need more maps!"
            self.mid_block = Basic2DEncoderBlock(
                in_channels=in_channels[-1],
                out_channels=mid_block_channels,
            )
            if use_checkpointing:
                assert fairscale is not None, "could not import fairscale library!"
                self.mid_block = fairscale.nn.checkpoint_wrapper(self.mid_block)
            smp_base.model.init.initialize_decoder(self.mid_block)
            prev_channels = mid_block_channels
        else:
            self.mid_block = None
            prev_channels = in_channels[-1]
        self.in_channels = in_channels
        self.mid_block_channels = mid_block_channels
        used_in_channels = in_channels[1:-1]  # decoders skip the original input + last pooled featmap
        if not block_out_channels:
            # auto-deduced: we'll use all the input channels as-is in reverse order!
            block_out_channels = [ch for ch in reversed(used_in_channels)]
        else:
            assert all([ch is not None and ch > 0 for ch in block_out_channels]), \
                "invalid channel count for decoder block outputs (should all be specified!)"
            assert not use_skip_connections or len(block_out_channels) >= len(used_in_channels), \
                "number of decoder blocks too small for the expected feature map count"
        assert all([c > 0 for c in block_out_channels]), "all ch counts should be specified!"
        self.block_out_channels = block_out_channels
        self.head_class_count = head_class_count

        blocks = []
        for block_idx, block_out_ch in enumerate(self.block_out_channels):
            if not use_skip_connections or block_idx >= len(used_in_channels):
                skip_channels = 0
            else:
                skip_channels = used_in_channels[-(block_idx + 1)]
            assert not self.head_class_count or block_out_ch > self.head_class_count, \
                "the decoder is too deep! (there are too few channels left for the head complexity)"
            block = Basic2DDecoderBlock(
                prev_channels=prev_channels,
                upsampl_channels=block_out_ch,
                skip_channels=skip_channels,
                out_channels=block_out_ch,
                coordconv=coordconv,
                attention=attention,
            )
            if use_checkpointing:
                assert fairscale is not None, "could not import fairscale library!"
                block = fairscale.nn.checkpoint_wrapper(block)
            blocks.append(block)
            prev_channels = block_out_ch
        self.blocks = torch.nn.ModuleList(blocks)
        smp_base.model.init.initialize_decoder(self.blocks)

        self.head = None
        if head_class_count is not None:
            assert prev_channels > head_class_count
            self.head = torch.nn.Conv2d(
                in_channels=prev_channels,
                out_channels=head_class_count,
                kernel_size=(1, 1),
            )
            smp_base.model.init.initialize_head(self.head)

        self.use_checkpointing = use_checkpointing
        self.use_skip_connections = use_skip_connections

    def forward(self, feat_maps):
        """Decodes a latent and a stack of feature maps into a class score map."""
        if self.use_skip_connections:
            assert len(feat_maps) == len(self.in_channels)
            assert all([fm.shape[1] == ch for fm, ch in zip(feat_maps, self.in_channels)])
            # first off, we'll discard the 1st feature map, it should be the input of the encoder
            # (it's a bit too low-level to be useful in this kind of decoder... to be tested?)
            feat_maps = feat_maps[1:]
            # we will also discard the last feature map, which is going to be fed as the 1st input
            skip_maps = feat_maps[::-1][1:]  # note: we reverse the featmap order for decoding!
        else:
            assert len(feat_maps) == 1 and feat_maps[0].shape[1] == self.in_channels[-1]
            skip_maps = []
        # next, forward the last feat map (bottleneck output) through the mid block
        if self.mid_block is not None:
            x = self.mid_block(feat_maps[-1])
        else:  # if there is no mid block, use the same feature map as-is
            x = feat_maps[-1]
        # iterate through all decoder blocks to slowly build the output map
        for block_idx, block in enumerate(self.blocks):
            if self.use_skip_connections and block_idx < len(skip_maps):
                skip = skip_maps[block_idx]
            else:
                skip = None
            x = block(x, skip)
        if self.head is not None:
            x = self.head(x)
        return x

    @property
    def block_count(self) -> int:
        """Returns the number of decoder blocks in this module."""
        return len(self.blocks)


class UNet(model_base.BaseSegmModel):
    """U-Net implementation. Offers a variety of encoder/decoder block configurations.

    The vanilla version includes batchnorm and transposed conv2d layers for upsampling. Coordinate
    Convolutions (CoordConv) can also be toggled on if requested.
    """

    def __init__(
        self,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates+logs model hyperparameters and sets up the model."""
        super().__init__(hyper_params)
        self.encoder, self.decoder = self._build_model(
            hyper_params=hyper_params,
            encoder_input_channels=self._get_expected_input_channel_count(),
            decoder_output_channels=self.head_class_count,
        )
        self.warned_bad_input_size_power2 = False

    @staticmethod
    def _get_model_block_types(
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ) -> typing.Tuple[typing.AnyStr, typing.AnyStr]:
        """Returns the encoder/decoder block types that will be instantiated in this unet."""
        # look for older (deprecated) hyperparameters, and use them if available...
        if "unet_encoder_type" in hyper_params:
            assert "encoder_type" not in hyper_params, "cannot provide old + new param names!"
            encoder_type = hyper_params["unet_encoder_type"]
        else:
            assert "encoder_type" in hyper_params, "missing required hyperparam 'encoder_type'"
            encoder_type = hyper_params["encoder_type"]
        assert encoder_type in SUPPORTED_ENCODER_TYPES, f"unexpected encoder type: {encoder_type}"
        # same thing for the decoder...
        if "unet_decoder_type" in hyper_params:
            assert "decoder_type" not in hyper_params, "cannot provide old + new param names!"
            decoder_type = hyper_params["unet_decoder_type"]
        else:
            assert "decoder_type" in hyper_params, "missing required hyperparam 'decoder_type'"
            decoder_type = hyper_params["decoder_type"]
        assert decoder_type in SUPPORTED_DECODER_TYPES, f"unexpected decoder type: {decoder_type}"
        # log the actual types that we'll be using
        fake_hparam_dict = dict(encoder_type=encoder_type, decoder_type=decoder_type)
        hp_utils.log_hp(list(fake_hparam_dict.keys()), fake_hparam_dict)
        return encoder_type, decoder_type

    @staticmethod
    def _get_encoder_info(
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ) -> typing.Tuple[int, typing.List[typing.Optional[int]]]:
        """Returns the block count and channel counts for the encoder."""
        # note: this function is essentially used for backward compatibility with older configs
        if "encoder_block_count" in hyper_params:
            encoder_block_count = hyper_params["encoder_block_count"]
            assert encoder_block_count > 0, f"invalid encoder block count! ({encoder_block_count})"
            if "encoder_block_channels" in hyper_params:
                encoder_block_channels = hyper_params["encoder_block_channels"]
                assert isinstance(encoder_block_channels, list), "unexpected type for channel list"
            else:
                encoder_block_channels = [None] * encoder_block_count  # will be auto-deduced
        else:
            assert "encoder_block_channels" in hyper_params, \
                "missing both encoder block count and channel list, should specify at least one!"
            encoder_block_channels = hyper_params["encoder_block_channels"]
            assert isinstance(encoder_block_channels, list), "unexpected type for channel list"
            assert len(encoder_block_channels) > 0, "invalid encoder channel list!"
            encoder_block_count = len(encoder_block_channels)
        assert len(encoder_block_channels) == encoder_block_count, \
            "mismatch between specified block count + ch list size!"
        assert all([not ch or ch > 0 for ch in encoder_block_channels]), "got invalid ch count!"
        fake_hparam_dict = dict(
            encoder_block_count=encoder_block_count, encoder_block_channels=encoder_block_channels,
        )
        hp_utils.log_hp(list(fake_hparam_dict.keys()), fake_hparam_dict)
        return encoder_block_count, encoder_block_channels

    @staticmethod
    def _fill_encoder_ch_counts_if_needed(
        encoder_block_channels: typing.List[typing.Optional[int]],
        mid_block_channels: typing.Optional[int],
        decoder_block_channels: typing.List[int],
    ) -> typing.List[int]:
        """Fills in the (possibly None) values of the encoder channel count list."""
        assert len(encoder_block_channels) > 0
        if not any([not ch or ch < 0 for ch in encoder_block_channels]):
            return encoder_block_channels  # nothing to do!
        # otherwise, we'll have to automatically deduce the ch counts one layer at a time...
        if not mid_block_channels or mid_block_channels < 0:
            assert len(decoder_block_channels) > 0
            prev_ch = decoder_block_channels[0] * 2  # if there is no mid block, check the decoder
        else:
            prev_ch = mid_block_channels  # if the mid block is defined, use it
        encoder_block_channels = copy.deepcopy(encoder_block_channels)  # to avoid ref issues...
        # now, iterate from the bottleneck back and fill in the missing values
        for curr_block_idx in reversed(range(len(encoder_block_channels))):
            curr_ch = encoder_block_channels[curr_block_idx]
            if not curr_ch or curr_ch < 0:
                encoder_block_channels[curr_block_idx] = prev_ch // 2
            prev_ch = encoder_block_channels[curr_block_idx]
        return encoder_block_channels

    @staticmethod
    def _build_model(
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
        encoder_input_channels: int,
        decoder_output_channels: typing.Optional[int],
    ) -> typing.Tuple[torch.nn.Module, torch.nn.Module]:
        """Builds and returns the PyTorch-compatible encoder-decoder model pair."""
        # set some of the new-but-maybe-missing default parameters for backward compatibility...
        encoder_type, decoder_type = UNet._get_model_block_types(hyper_params)
        hyper_params["use_skip_connections"] = hyper_params.get("use_skip_connections", True)
        encoder_block_count, encoder_block_channels = UNet._get_encoder_info(hyper_params)
        hp_utils.check_and_log_hp(
            names=[
                "coordconv",
                "mid_block_channels",  # 0 if we don't want a mid block; 256-512 for vanilla unet
                "decoder_block_channels",
                "decoder_attention_type",
                "use_skip_connections",
                "use_checkpointing",
            ],
            hps=hyper_params,
        )
        coordconv = hyper_params["coordconv"]
        mid_block_channels = hyper_params["mid_block_channels"]
        decoder_block_channels = hyper_params["decoder_block_channels"]
        decoder_block_channels = \
            hp_utils.get_array_from_input_that_could_be_a_string(decoder_block_channels)
        decoder_attention_type = hyper_params["decoder_attention_type"]
        use_skip_connections = hyper_params["use_skip_connections"]
        use_checkpointing = hyper_params["use_checkpointing"]

        if encoder_type == "vanilla":
            # build a classic (simple) CNN based only on stacks of Conv2d-BN-ReLU layers
            encoder_block_channels = UNet._fill_encoder_ch_counts_if_needed(
                encoder_block_channels=encoder_block_channels,
                mid_block_channels=mid_block_channels,
                decoder_block_channels=decoder_block_channels,
            )
            encoder = Basic2DEncoder(
                in_channels=encoder_input_channels,
                block_out_channels=encoder_block_channels,
                coordconv=coordconv,
            )
            assert encoder.block_count == encoder_block_count
        elif encoder_type in ["CustomResNet", "custom-resnet", "resnet"]:
            # grab all required hyperparameters and redirect everything to the custom resnet encoder
            # TODO: we'll need to update the custom resnet class if we want to do coordconv in it...
            # (or use the auto-conv2d-swap utility in coordconv module; not tested in a while!)
            assert not coordconv, "missing implementation! (see comment above)"
            assert mid_block_channels <= 0, "the mid block is skipped for the custom resnet encoder"
            assert all([ch > 0 for ch in encoder_block_channels]), \
                "custom resnet encoder must be defined with specific ch counts for each stage!"
            assert "extra_encoder_params" in hyper_params, \
                "missing 'extra_encoder_params' field for the custom resnet impl in unet definition"
            extra_encoder_params = hyper_params["extra_encoder_params"]
            assert isinstance(extra_encoder_params, dict), "unexpected extra encoder params field type"
            hp_utils.log_hp(["extra_encoder_params"], hyper_params)
            expected_extra_hparams = ["block", "layers"]
            assert all([k in extra_encoder_params for k in expected_extra_hparams]), \
                "missing one or more mandatory extra encoder hparams in the unet encoder definition"
            unexpected_extra_hparams = ["channels", "in_channels"]
            assert all([k not in extra_encoder_params for k in unexpected_extra_hparams]), \
                "found one or more unnecessary extra encoder hparams in the unet encoder definition"
            encoder = custom_resnet.ResNetEncoder(
                channels=encoder_block_channels,
                in_channels=encoder_input_channels,
                **extra_encoder_params,
            )
        elif encoder_type in smp.encoders.encoders:
            # TODO: we'll need to derive a bunch of classes if we want to do coordconv in those...
            # (or use the auto-conv2d-swap utility in coordconv module; not tested in a while!)
            assert not coordconv, "missing implementation! (see comment above)"
            assert mid_block_channels <= 0, "the mid block is skipped for smp backbones"
            assert all([not ch or ch < 0 for ch in encoder_block_channels]), \
                "smp backbones cannot be combined with specific ch counts in the encoder blocks!"
            encoder = smp.encoders.get_encoder(
                name=encoder_type,
                in_channels=encoder_input_channels,
                depth=encoder_block_count,
                weights=None,  # TODO: should we init with pretrained weights here?
            )
        else:
            raise NotImplementedError  # TODO: add more encoders here!

        if decoder_type == "vanilla":
            decoder = Basic2DDecoder(
                in_channels=encoder.out_channels,
                mid_block_channels=mid_block_channels,
                block_out_channels=decoder_block_channels,
                head_class_count=decoder_output_channels,
                coordconv=coordconv,
                attention=decoder_attention_type,
                use_checkpointing=use_checkpointing,
                use_skip_connections=use_skip_connections,
            )
        else:
            raise NotImplementedError  # TODO: add more decoders here!

        if use_checkpointing:
            assert fairscale is not None, "could not import fairscale library!"
            encoder = fairscale.nn.checkpoint_wrapper(encoder)

        return encoder, decoder

    @profile
    def forward(self, x):
        """Forwards the provided tensor through the encoder/mid_block/decoder modules."""
        if torch.is_grad_enabled() and self.use_checkpointing:
            # Only set grad equals true on input if we are in a training phase.
            # this is required by fairscale's checkpointing method.
            x.requires_grad_(True)
        self._check_input_tensor_pow2_size(x)
        feat_maps = self.encoder(x)
        out = self.decoder(feat_maps)
        return out

    @profile
    def _generic_step(
        self,
        batch: typing.Any,
        batch_idx: int,
        evaluator: metrics_base.EvaluatorBase,
    ) -> typing.Tuple[typing.Any, torch.Tensor, typing.Dict[typing.AnyStr, float]]:
        """Runs the prediction + evaluation step for training/validation/testing."""
        input_tensor = self._prepare_input_features(batch)
        preds = self(input_tensor)  # calls the forward pass of the model
        assert self.segm_mask_field_name in batch, "forgot to generate the segmentation masks in preproc?"
        targets = batch[self.segm_mask_field_name].long()
        loss = self.loss_fn(preds, targets)
        preds = preds.detach()  # anything else done with the preds should not affect the model
        metrics = evaluator.ingest(batch, batch_idx, preds)
        return preds, loss, metrics

    def _check_input_tensor_pow2_size(self, x):
        """Checks whether we need to warn about bad image size or not."""
        if not self.warned_bad_input_size_power2 and len(x.shape) == 4:
            if not math.log(x.shape[-1], 2).is_integer() or not math.log(x.shape[-2], 2).is_integer():
                self.warned_bad_input_size_power2 = True
                logger.warning("unet input size should be power of 2 (e.g. 256x256, 512x512, ...)")
