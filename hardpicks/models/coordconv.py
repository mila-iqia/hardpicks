import inspect
import typing

import torch
import torch.nn


def get_coords_map(
    height: int,
    width: int,
    centered: bool = True,
    normalized: bool = True,
    noise_std: bool = None,
    dtype: typing.Optional[typing.Type] = torch.float32,
) -> torch.Tensor:
    """Returns a H+W intrinsic coordinates map tensor (shape=2xHxW)."""
    assert isinstance(height, int) and height > 0
    assert isinstance(width, int) and width > 0
    x = torch.arange(width, dtype=dtype)
    y = torch.arange(height, dtype=dtype)
    if centered:
        x -= (width - 1) // 2
        y -= (height - 1) // 2
    if normalized:
        x /= width - 1
        y /= height - 1
    y, x = torch.meshgrid(y, x)
    if noise_std is not None:
        assert isinstance(noise_std, (int, float)) and noise_std >= 0, "invalid noise_std value"
        x = torch.normal(mean=x, std=noise_std)
        y = torch.normal(mean=y, std=noise_std)
    return torch.stack([y, x])


def add_radius_channel_to_coords_map(
    coords_map: torch.Tensor,
    target_pt: typing.Optional[typing.Tuple[float, float]] = None,
) -> torch.Tensor:
    """Concatenates and returns a new tensor with a radius (distance-to-coord) channel."""
    assert isinstance(coords_map, torch.Tensor) and len(coords_map.shape) == 3
    coord_ch, height, width = coords_map.shape
    assert coord_ch == 2, "unexpected coord channel count (should be only be 2, i.e. x + y)"
    if target_pt is None:
        target_pt = coords_map[:, (height - 1) // 2, (width - 1) // 2]
    assert len(target_pt) == 2
    radius = torch.sqrt(
        torch.pow(coords_map[0, :, :] - target_pt[0], 2)
        + torch.pow(coords_map[1, :, :] - target_pt[1], 2)
    )
    return torch.cat([coords_map, radius.unsqueeze(0)], dim=0)


class AddCoords(torch.nn.Module):
    """Creates a torch-compatible layer that adds intrinsic coordinate layers to input tensors."""

    def __init__(
        self,
        centered=True,
        normalized=True,
        noise_std=None,
        radius_channel=False,
        scale=None,
    ):
        """Stores layer parameters. This depends on no extra actual module."""
        super().__init__()
        self.centered = centered
        self.normalized = normalized
        self.noise_std = noise_std
        self.radius_channel = radius_channel
        assert scale is None or isinstance(scale, (int, float)), "invalid scale type"
        self.scale = scale

    def forward(self, in_tensor):
        """Adds coordinate channels to the provided tensor and returns it."""
        batch_size, channels, height, width = in_tensor.shape
        coords_map = get_coords_map(height, width, self.centered, self.normalized, self.noise_std)
        if self.scale is not None:
            coords_map *= self.scale
        if self.radius_channel:
            coords_map = add_radius_channel_to_coords_map(coords_map)
        coords_map = coords_map.repeat(batch_size, 1, 1, 1)
        dev = in_tensor.device
        out = torch.cat([in_tensor, coords_map.to(dev)], dim=1)
        return out


class CoordConv2d(torch.nn.Module):
    """CoordConv-equivalent of torch's default Conv2d model layer.

    .. seealso::
        | Liu et al., An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution`
          <https://arxiv.org/abs/1807.03247>`_ [arXiv], 2018.
    """

    def __init__(
        self,
        in_channels,
        *args,
        centered=True,
        normalized=True,
        noise_std=None,
        radius_channel=False,
        scale=None,
        **kwargs,
    ):
        """Creates an `AddCoords` layer and a `Conv2d` layer based on the given args."""
        super().__init__()
        self.addcoord = AddCoords(
            centered=centered,
            normalized=normalized,
            noise_std=noise_std,
            radius_channel=radius_channel,
            scale=scale,
        )
        extra_ch = 3 if radius_channel else 2
        self.conv = torch.nn.Conv2d(in_channels + extra_ch, *args, **kwargs)

    def forward(self, in_tensor):
        """Forwards the provided tensor (plus extra coordinate channels)through the conv layer."""
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose2d(torch.nn.Module):
    """CoordConv-equivalent of torch's default ConvTranspose2d model layer.

    .. seealso::
        | Liu et al., An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution`
          <https://arxiv.org/abs/1807.03247>`_ [arXiv], 2018.
    """

    def __init__(
        self,
        in_channels,
        *args,
        centered=True,
        normalized=True,
        noise_std=None,
        radius_channel=False,
        scale=None,
        **kwargs,
    ):
        """Creates an `AddCoords` layer and a `ConvTranspose2d` layer based on the given args."""
        super().__init__()
        self.addcoord = AddCoords(
            centered=centered,
            normalized=normalized,
            noise_std=noise_std,
            radius_channel=radius_channel,
            scale=scale,
        )
        extra_ch = 3 if radius_channel else 2
        self.conv = torch.nn.ConvTranspose2d(in_channels + extra_ch, *args, **kwargs)

    def forward(self, in_tensor):
        """Forwards the provided tensor (plus extra coordinate channels)through the conv layer."""
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


def make_conv2d(
    *args,
    coordconv=False,
    **kwargs,
):
    """Creates a 2D convolution layer with optional CoordConv support."""
    if coordconv:
        return CoordConv2d(
            *args,
            **kwargs,
        )
    else:
        # just a failsafe to make sure none of the coordconv-related keyword are forwarded
        argspec = inspect.getfullargspec(torch.nn.Conv2d)
        coordconv_args = ["centered", "normalized", "noise_std", "radius_channel", "scale"]
        assert all([ccarg not in argspec.args for ccarg in coordconv_args])
        valid_kwargs = {key: val for key, val in kwargs.items() if key not in coordconv_args}
        return torch.nn.Conv2d(*args, **valid_kwargs)
