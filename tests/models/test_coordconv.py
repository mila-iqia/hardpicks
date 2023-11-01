import mock
import numpy as np
import pytest
import torch

import hardpicks.models.coordconv as cc


@pytest.mark.parametrize(
    "rows,cols", [(1, 1), (4, 4), (12, 24), (32, 16), (99, 100)],
)
def test_coords_map_raw(rows, cols):
    map = cc.get_coords_map(height=rows, width=cols, centered=False, normalized=False, noise_std=None)
    assert map.ndim == 3 and map.shape[0] == 2 and map.shape[1] == rows and map.shape[2] == cols
    for row_idx in range(rows):
        for col_idx in range(cols):
            # just test that indices match as expected in each row/column
            assert (map[:, row_idx, col_idx] == torch.Tensor((row_idx, col_idx))).all()


@pytest.mark.parametrize(
    "rows,cols", [(5, 5), (9, 9), (17, 25), (41, 20), (100, 100)],
)
def test_coords_map_centered(rows, cols):
    map = cc.get_coords_map(height=rows, width=cols, centered=True, normalized=False, noise_std=None)
    assert map.ndim == 3 and map.shape[0] == 2 and map.shape[1] == rows and map.shape[2] == cols
    center = ((rows - 1) // 2, (cols - 1) // 2)
    assert (map[:, center[0], center[1]] == torch.Tensor((0, 0))).all()
    for row_offset in range(-1, 2):
        for col_offset in range(-1, 2):
            # in this case, the indices around the center should match monotonically with offsets
            vals = map[:, center[0] + row_offset, center[1] + col_offset]
            assert (vals == torch.Tensor((row_offset, col_offset))).all()


@pytest.mark.parametrize(
    "rows,cols", [(5, 5), (10, 10), (100, 100)],
)
def test_coords_map_normalized(rows, cols):
    raw = cc.get_coords_map(height=rows, width=cols, centered=False, normalized=False, noise_std=None)
    map = cc.get_coords_map(height=rows, width=cols, centered=False, normalized=True, noise_std=None)
    assert map.ndim == 3 and map.shape[0] == 2 and map.shape[1] == rows and map.shape[2] == cols
    # when normalized, the map should just be a min/max-'d version of the original
    max_y, max_x = (raw[0].max().item(), raw[1].max().item())
    assert max_y == rows - 1 and max_x == cols - 1
    raw[0] /= max_y
    raw[1] /= max_x
    assert (map == raw).all()


def test_coords_map_noise_std():
    noise_std, rows, cols = 5, 10001, 10000
    raw = cc.get_coords_map(height=rows, width=cols, centered=False, normalized=False, noise_std=None)
    map = cc.get_coords_map(height=rows, width=cols, centered=False, normalized=False, noise_std=noise_std)
    assert map.ndim == 3 and map.shape[0] == 2 and map.shape[1] == rows and map.shape[2] == cols
    # here, we'll just verify that the noise is indeed applied and it is gaussian
    assert not np.allclose(raw, map)
    diff = (map - raw).numpy()
    assert np.isclose(diff.mean(), 0, rtol=0, atol=0.01)
    assert np.isclose(diff.std(), noise_std, rtol=0, atol=0.01)


def test_add_vanilla_and_scaled_coords():
    batch_size, orig_channels, rows, cols = 16, 6, 32, 32
    in_tensor = torch.randn(batch_size, orig_channels, rows, cols)
    vanilla_add_coords_layer = cc.AddCoords(
        centered=False, normalized=False, noise_std=None,
        radius_channel=False, scale=None,
    )
    out_tensor = vanilla_add_coords_layer(in_tensor)
    # the layer should have added two new channels but left everything else intact
    assert out_tensor.ndim == 4
    assert out_tensor.shape[0] == batch_size
    assert out_tensor.shape[1] == orig_channels + 2
    assert out_tensor.shape[2] == rows
    assert out_tensor.shape[3] == cols
    assert (out_tensor[:, :orig_channels, :, :] == in_tensor).all()
    batch_coords = out_tensor[:, orig_channels:, :, :].numpy()
    # all coordinate channels should be exactly the same along the batch dimension
    assert np.equal(batch_coords, batch_coords[0]).all()
    raw = cc.get_coords_map(height=rows, width=cols, centered=False, normalized=False, noise_std=None)
    # ... and they should match with the 'raw' coordinate map we get manually
    assert np.equal(batch_coords, raw.numpy()).all()
    # finally, check that scaling works on all coord values
    scale_factor = 2
    scaled_add_coords_layer = cc.AddCoords(
        centered=False, normalized=False, noise_std=None,
        radius_channel=False, scale=scale_factor,
    )
    out_scaled_tensor = scaled_add_coords_layer(in_tensor)
    assert out_tensor.shape == out_scaled_tensor.shape
    assert (out_scaled_tensor[:, :orig_channels, :, :] == in_tensor).all()
    scaled_batch_coords = out_scaled_tensor[:, orig_channels:, :, :].numpy()
    assert (batch_coords * scale_factor == scaled_batch_coords).all()


@pytest.mark.parametrize(
    "rows,cols", [(5, 5), (9, 9), (17, 25), (41, 20), (100, 100)],
)
def test_add_coords_and_radius(rows, cols):
    batch_size, orig_channels = 16, 6
    in_tensor = torch.randn(batch_size, orig_channels, rows, cols)
    add_coords_and_radius_layer = cc.AddCoords(
        centered=False, normalized=False, noise_std=None,
        radius_channel=True, scale=None,
    )
    out_tensor = add_coords_and_radius_layer(in_tensor)
    # the layer should have added three new channels but left everything else intact
    assert out_tensor.ndim == 4
    assert out_tensor.shape[0] == batch_size
    assert out_tensor.shape[1] == orig_channels + 3
    assert out_tensor.shape[2] == rows
    assert out_tensor.shape[3] == cols
    assert (out_tensor[:, :orig_channels, :, :] == in_tensor).all()
    batch_coords = out_tensor[:, orig_channels:(orig_channels + 2), :, :].numpy()
    batch_radius = out_tensor[:, orig_channels + 2, :, :].numpy()
    # all coordinate channels should be exactly the same along the batch dimension
    assert np.equal(batch_coords, batch_coords[0]).all()
    assert np.equal(batch_radius, batch_radius[0]).all()
    # ... and they should match with the 'raw' coordinate map we get manually
    raw = cc.get_coords_map(height=rows, width=cols, centered=False, normalized=False, noise_std=None)
    assert np.equal(batch_coords, raw.numpy()).all()
    # lastly, just check that the center is indeed the min value in the radius map
    expected_center = (rows - 1) // 2, (cols - 1) // 2
    assert not (batch_radius == 0).all()
    assert (batch_radius[:, expected_center[0], expected_center[1]] == 0).all()


def mocked_addcoord_layer_setter(*args, **kwargs):
    assert len(args) == 3 and args[1] == "addcoord"
    modules = args[0].__dict__.get('_modules')
    if "addcoord" in modules:
        modules["addcoord"] = args[2]


def test_conv_wrappers():
    batch_size, orig_channels, rows, cols = 16, 6, 32, 32
    in_tensor = torch.randn(batch_size, orig_channels, rows, cols)
    # here, we'll just check that the add coords layer exists and is called
    conv2d = cc.CoordConv2d(
        in_channels=orig_channels,
        out_channels=orig_channels * 2,
        kernel_size=3,
        padding=1,
    )
    assert isinstance(conv2d.addcoord, cc.AddCoords)
    with mock.patch.object(torch.nn.Module, "__setattr__", new=mocked_addcoord_layer_setter) as _, \
            mock.patch.object(conv2d, "addcoord", wraps=conv2d.addcoord) as fake_addcoords:
        out_tensor = conv2d(in_tensor)
        assert fake_addcoords.call_count == 1
        assert out_tensor.shape == (batch_size, orig_channels * 2, rows, cols)

    convtransp2d = cc.CoordConvTranspose2d(
        in_channels=orig_channels,
        out_channels=orig_channels // 2,
        kernel_size=2,
        stride=2,
    )
    assert isinstance(convtransp2d.addcoord, cc.AddCoords)
    with mock.patch.object(torch.nn.Module, "__setattr__", new=mocked_addcoord_layer_setter) as _, \
            mock.patch.object(convtransp2d, "addcoord", wraps=convtransp2d.addcoord) as fake_addcoords:
        out_tensor = convtransp2d(in_tensor)
        assert fake_addcoords.call_count == 1
        assert out_tensor.shape == (batch_size, orig_channels // 2, rows * 2, cols * 2)


def test_make_conv2d_wrapper():
    # we want to be able to call that function with extra args without fail based on coordconv flag
    default_conv2d_args = dict(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    layer = cc.make_conv2d(
        **default_conv2d_args,
        coordconv=False,
        # and provide one extra coordconv-only arg to remove in the wrapper...
        radius_channel=True,
    )
    assert isinstance(layer, torch.nn.Conv2d)  # we asked for no coordconv, should not be wrapped
    # try again with the flag turned on and make sure this time we get the right object
    layer = cc.make_conv2d(
        **default_conv2d_args,
        coordconv=True,
        radius_channel=True,
    )
    assert isinstance(layer, cc.CoordConv2d)
    assert layer.conv.in_channels == 19
