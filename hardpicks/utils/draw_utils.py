import typing

import cv2 as cv
import einops
import numpy as np
import torch

import hardpicks.data.fbp.constants as fbp_consts
import hardpicks.models.constants as model_consts

seismic_ampl_mean, seismic_ampl_std = 0.0, 0.5  # for normalization-before-display purposes only
default_pad_color = (218, 224, 237)  # BGR color of the default padding to use in displays
default_text_color = (32, 26, 26)  # BGR color of the text to render in the displays


def fig2array(fig) -> np.ndarray:  # pragma: no cover
    """Transforms a pyplot figure into a numpy-compatible RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf


def get_displayable_image(
    array: np.ndarray,
    grayscale: bool = False,
    mask: typing.Optional[np.ndarray] = None,
) -> np.ndarray:  # pragma: no cover
    """Returns a 'displayable' image that has been normalized and padded to three channels."""
    assert array.ndim in [2, 3], "unexpected input array dim count"
    if array.ndim == 3:
        if array.shape[2] == 2:
            array = np.dstack((array, array[:, :, 0]))
        elif array.shape[2] > 3:
            array = array[..., :3]
    image = cv.normalize(array, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U, mask)
    if grayscale and array.ndim == 3 and array.shape[2] != 1:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif not grayscale and (array.ndim == 2 or array.shape[2] == 1):
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return image


def get_displayable_heatmap(array: np.ndarray,) -> np.ndarray:  # pragma: no cover
    """Returns a 'displayable' array that has been min-maxed and mapped to BGR triplets."""
    if array.ndim != 2:
        array = np.squeeze(array)
    assert array.ndim == 2, "unexpected input array dim count"
    array = cv.normalize(array, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    heatmap = cv.applyColorMap(array, cv.COLORMAP_VIRIDIS)
    return heatmap


def get_displayable_binary_mask(mask: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Returns a 'displayable' version of a binary segmentation mask (with optional dontcare)."""
    assert mask.ndim == 2 and mask.dtype in [np.int32, np.int16]
    mask[mask > 0] = 255  # full white pixels for positive class
    mask[mask == model_consts.DONTCARE_SEGM_MASK_LABEL] = 64  # dark gray for dontcare pixels
    mask = cv.cvtColor(mask.astype(np.uint8), cv.COLOR_GRAY2BGR)
    return mask


def draw_big_text_at_top_right(
    image: np.ndarray,
    text: typing.AnyStr,
    font: int = cv.FONT_HERSHEY_PLAIN,
    font_scale: float = 4.0,
    font_thickness: int = 6,
    font_color: typing.Tuple[int, int, int] = (224, 64, 64),  # blue text
) -> np.ndarray:  # pragma: no cover
    """Draws the given text (bigly!) at the top right of the given image and returns it."""
    assert (
        image.ndim == 3 and image.shape[-1] == 3
    ), "input image should be BGR already!"
    (text_width, text_height), text_baseline = cv.getTextSize(
        text, font, font_scale, font_thickness
    )
    text_origin = (image.shape[1] - 15 - text_width, text_height + 15)
    image = cv.putText(
        image, text, text_origin, font, font_scale, (255, 255, 255), font_thickness + 2
    )
    image = cv.putText(
        image, text, text_origin, font, font_scale, font_color, font_thickness
    )
    return image


def draw_first_break_points(
    image: np.ndarray,
    pred_first_break_labels: typing.Optional[np.ndarray],
    target_first_break_labels: typing.Optional[np.ndarray],
    valid_trace_mask: np.ndarray,
    bad_first_break_mask: np.ndarray,
    outlier_first_break_mask: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    """Draws the target and predicted first break points on the gather image."""
    # predicted points are all drawn in blue;
    # target points are in red (if invalid), orange (if outliers), or green (if valid)
    assert (
        image.ndim == 3 and image.shape[-1] == 3
    ), "input image should be BGR already!"
    assert valid_trace_mask.ndim == 1 and len(valid_trace_mask) == image.shape[0]
    if pred_first_break_labels is not None:
        assert (
            pred_first_break_labels.ndim == 1
            and len(pred_first_break_labels) == image.shape[0]
        )
    if target_first_break_labels is not None:
        assert (
            target_first_break_labels.ndim == 1
            and len(target_first_break_labels) == image.shape[0]
        )
    assert valid_trace_mask.shape == bad_first_break_mask.shape
    assert valid_trace_mask.shape == outlier_first_break_mask.shape
    for trace_idx in np.where(valid_trace_mask)[0]:
        if target_first_break_labels is not None:
            curr_target_fb_sample_idx = min(
                max(target_first_break_labels[trace_idx], 0), image.shape[1]
            )
            curr_target_fb_color = (
                16,
                232,
                16,
            )  # nice green for all valid target points
            if bad_first_break_mask[trace_idx]:
                curr_target_fb_color = (
                    16,
                    16,
                    212,
                )  # red for all invalid target points
            elif outlier_first_break_mask[trace_idx]:
                curr_target_fb_color = (
                    4,
                    142,
                    232,
                )  # orange for all outlier target points
            image = cv.circle(
                image,
                (curr_target_fb_sample_idx, trace_idx),
                1,
                curr_target_fb_color,
                thickness=-1,
            )
        if pred_first_break_labels is not None:
            curr_pred_fb_sample_idx = (
                image.shape[1] - 1
            )  # send all invalid points at max pixel by default
            if (
                pred_first_break_labels[trace_idx]
                > fbp_consts.BAD_FIRST_BREAK_PICK_INDEX
            ):
                curr_pred_fb_sample_idx = pred_first_break_labels[trace_idx]
            curr_pred_fb_color = (244, 16, 16)  # blue for all predicted points
            image = cv.circle(
                image,
                (curr_pred_fb_sample_idx, trace_idx),
                1,
                curr_pred_fb_color,
                thickness=-1,
            )
    return image


def generate_gather_image_from_batch(
    batch: typing.Dict, gather_idx: int, draw_prior: bool = False,
) -> np.ndarray:  # pragma: no cover
    """Generates and returns the image of a 2d line shot gather."""
    samples_mean, samples_stdev = (
        torch.mean(batch["samples"].float()),
        torch.std(batch["samples"].float()),
    )
    samples = (
        ((batch["samples"][gather_idx].float() - samples_mean) / samples_stdev)
        .cpu()
        .numpy()
    )
    # we'll clip the samples to 2 stdevs (it'll highlight 95% of the variations properly)
    samples = np.clip(samples, -2, 2)
    # next, extract the mask of not-padded samples to highlight only the right columns/rows
    orig_trace_count = batch["trace_count"][gather_idx].item()
    orig_sample_count = batch["sample_count"][gather_idx].item()
    not_padded_samples_mask = np.zeros_like(samples, dtype=np.uint8)
    not_padded_samples_mask[0:orig_trace_count, 0:orig_sample_count] = 255
    # normalize the clipped samples into a proper 0-255 range for RGB rendering
    image = get_displayable_image(samples, mask=not_padded_samples_mask)
    if draw_prior and "first_break_prior" in batch:
        prior = batch["first_break_prior"][gather_idx].cpu().numpy()
        prior = (np.clip(prior, 0, 1) * 255).astype(np.uint8)
        prior_rgb = np.stack(
            (np.zeros_like(prior), np.zeros_like(prior), prior), axis=-1
        )
        image = cv.addWeighted(image, 0.5, prior_rgb, 0.5, 0)
    return image


def generate_seismic_slice_image_from_batch(
    batch: typing.Dict, image_idx: int,
) -> np.ndarray:  # pragma: no cover
    """Generates and returns the image of a 2d seismic slice."""
    # we do a quick mean-stdev normalization, clip to 95% of the spread, and return as-is
    seismic = batch["seismic_amplitude"][image_idx].float()
    seismic = np.clip(
        ((seismic - torch.mean(seismic)) / torch.std(seismic)).cpu().numpy(), -2, 2
    )
    orig_shape = batch["orig_slice_shape"][image_idx].cpu().numpy()
    not_padded_mask = np.zeros_like(seismic, dtype=np.uint8)
    not_padded_mask[0: orig_shape[0], 0: orig_shape[1]] = 255
    return get_displayable_image(seismic, mask=not_padded_mask)


def add_heatmap_on_base_image(
    base_image: np.ndarray, heat_values: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    """Adds a colorful heatmap on top of a base image."""
    assert base_image.ndim == 2 or base_image.ndim == 3
    if base_image.ndim == 2 or (base_image.ndim == 3 and base_image.shape[-1] == 1):
        base_image = cv.cvtColor(base_image, cv.COLOR_GRAY2BGR)
    assert base_image.ndim == 3 and base_image.shape[-1] == 3
    assert base_image.dtype == np.uint8
    heatmap = get_displayable_heatmap(heat_values)
    image = cv.addWeighted(base_image, 0.3, heatmap, 0.7, 0)
    return image


def get_cv_colormap_from_class_color_map(
    class_color_list: typing.Sequence[
        typing.Union[np.ndarray, typing.Tuple[int, int, int]]
    ],
    default_color: np.ndarray = np.array([0xFF, 0xFF, 0xFF], dtype=np.uint8),
) -> np.ndarray:  # pragma: no cover
    """Converts a list of color triplets into a 256-len array of color triplets for OpenCV."""
    assert (
        len(class_color_list) < 256
    ), "invalid class color list (should be less than 256 classes)"
    out_color_array = []
    for label_idx in range(256):
        if label_idx < len(class_color_list):
            assert (
                len(class_color_list[label_idx]) == 3
            ), f"invalid triplet for idx={label_idx}"
            out_color_array.append(
                class_color_list[label_idx][::-1]
            )  # RGB to BGR for opencv
        else:
            out_color_array.append(default_color)
    return np.asarray(out_color_array).astype(np.uint8)


def apply_cv_colormap(class_idx_map: np.ndarray, color_map: np.ndarray,) -> np.ndarray:
    """Applies the OpenCV color map onto the class label index map, channel by channel."""
    assert (
        model_consts.DONTCARE_SEGM_MASK_LABEL == -1
    ), "logic below only works for -1 dc label..."
    assert np.issubdtype(class_idx_map.dtype, np.integer)
    min_label_idx, max_label_idx, _, _ = cv.minMaxLoc(class_idx_map)
    assert min_label_idx >= -1 and max_label_idx < 255, "invalid label index range"
    class_idx_map = class_idx_map.astype(np.uint8)  # dontcare label becomes 255
    assert color_map.shape == (256, 3), "invalid color map shape"
    output = np.zeros((*class_idx_map.shape, 3), dtype=np.uint8)
    for ch_idx in range(3):
        output[..., ch_idx] = cv.applyColorMap(class_idx_map, color_map[..., ch_idx])[
            ..., ch_idx
        ]
    return output


def get_html_color_code(rgb: typing.Tuple[int, int, int]) -> str:
    """Returns the HTML (hex) color code given a tuple of R,G,B values."""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def get_2_by_2_display_grid(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
):
    """Display 4 images in a 2x2 padded grid."""
    # TODO: this "works", but seems a little bit fragile, especially around
    #   converting arrays to what cv can understand. Maybe there's a better way?
    list_images = []

    image_height = top_left.shape[0]
    image_width = top_left.shape[1]

    for image_array in [top_left, top_right, bottom_left, bottom_right]:
        assert image_array.shape[0] == image_height, "inconsistent height between images"
        assert image_array.shape[1] == image_width, "inconsistent width between images"

        if len(image_array.shape) == 2:
            image = get_displayable_image(image_array)
        elif len(image_array.shape) == 3:
            image = image_array
        else:
            raise ValueError("unexpected input array shape")
        list_images.append(image)

    # create separators between image panes
    ones = np.ones([image_height, int(0.05 * image_width)], dtype=np.uint8)
    vertical_separator_image = einops.repeat(ones, "h w -> h w c", c=3)

    top_row = cv.hconcat([list_images[0], vertical_separator_image, list_images[1]])
    bottom_row = cv.hconcat([list_images[2], vertical_separator_image, list_images[3]])

    row_height = top_row.shape[0]
    row_width = top_row.shape[1]
    ones = np.ones([int(0.05 * row_height), row_width], dtype=np.uint8)
    separator_row = einops.repeat(ones, "h w -> h w c", c=3)
    display = cv.vconcat([top_row, separator_row, bottom_row])
    return display


def convert_seismic_ampl_to_8bit(
    array: typing.Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """Converts a floating point array of seismic amplitudes into a normalized 8-bit format."""
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    assert array.dtype != np.uint8
    array = np.clip(((array - seismic_ampl_mean) / seismic_ampl_std), -2, 2)
    array = (((array + 2) / 4) * 255).astype(np.uint8)
    return array


def resize_nn(
    image: np.ndarray,
    zoom_factor: float = 1,
) -> np.ndarray:
    """Performs nearest-neighbor resampling of a BGR image with a given scaling factor."""
    assert image.ndim == 3
    if zoom_factor != 1:
        return cv.resize(
            src=image,
            dsize=(-1, -1),
            fx=zoom_factor, fy=zoom_factor,
            interpolation=cv.INTER_NEAREST,
        )
    return image


def render_seismic_patch(
    patch: np.ndarray,
    base_zoom_factor: float = 4,
    border_size: int = 1,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Converts a raw (not-yet-normalized) seismic amplitude patch into an 8-bit BGR image."""
    if patch.dtype != np.uint8:
        patch = convert_seismic_ampl_to_8bit(patch)
    if patch.ndim != 3:
        patch = cv.cvtColor(patch, cv.COLOR_GRAY2BGR)
    patch = cv.copyMakeBorder(
        src=patch,
        top=border_size, bottom=border_size, left=border_size, right=border_size,
        borderType=cv.BORDER_CONSTANT,
        value=pad_color,
    )
    return resize_nn(patch, base_zoom_factor)


def add_subtitle_to_image(
    image: np.ndarray,
    subtitle: str,
    extra_border_size: int = 0,
    extra_subtitle_padding: int = 2,
    scale: float = 2.0,
    thickness: typing.Optional[int] = 2,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Renders an image with a small subtitle string underneath it."""
    assert image.ndim == 3
    text_size, baseline = cv.getTextSize(
        text=subtitle,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=scale,
        thickness=thickness,
    )
    if text_size[0] > image.shape[1]:
        extra_x_padding = (text_size[0] - image.shape[1]) // 2
    else:
        extra_x_padding = 0
    image = cv.copyMakeBorder(
        src=image,
        top=extra_border_size,
        bottom=(text_size[1] + extra_border_size + extra_subtitle_padding * 2),
        left=extra_border_size + extra_x_padding,
        right=extra_border_size + extra_x_padding,
        borderType=cv.BORDER_CONSTANT,
        value=pad_color,
    )
    out_x = int(0.5 * image.shape[1] - text_size[0] // 2)
    out_y = image.shape[0] - extra_subtitle_padding
    cv.putText(
        img=image,
        text=subtitle,
        org=(out_x, out_y),  # X, Y, as expected by opencv
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=scale,
        color=default_text_color,
        thickness=thickness,
        lineType=None,
        bottomLeftOrigin=None,
    )
    return image


def append_image_to_string(
    image: np.ndarray,
    string: str,
    extra_border_size: int = 0,
    extra_string_padding: int = 20,
    scale: float = 3.0,
    thickness: typing.Optional[int] = 3,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Renders a text string followed by an image on its right with optional padding."""
    assert image.ndim == 3
    text_size, baseline = cv.getTextSize(
        text=string,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=scale,
        thickness=thickness,
    )
    assert text_size[1] < image.shape[0]
    image = cv.copyMakeBorder(
        src=image,
        top=extra_border_size,
        bottom=extra_border_size,
        left=extra_border_size + text_size[0] + extra_string_padding * 2,
        right=extra_border_size,
        borderType=cv.BORDER_CONSTANT,
        value=pad_color,
    )
    out_x = extra_string_padding
    out_y = (image.shape[0] + text_size[1]) // 2
    cv.putText(
        img=image,
        text=string,
        org=(out_x, out_y),  # X, Y, as expected by opencv
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=scale,
        color=default_text_color,
        thickness=thickness,
        lineType=None,
        bottomLeftOrigin=None,
    )
    return image


def vconcat_and_pad_if_needed(
    images: typing.Sequence[np.ndarray],
    extra_border_padding: int = 10,
    extra_vertical_padding: int = 20,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Concatenates a list of images vertically with optional auto-padding."""
    assert all([img.ndim == 3 for img in images])
    max_width = max([img.shape[1] for img in images])
    padded_images = []
    for img_idx, img in enumerate(images):
        req_padding = max_width - img.shape[1]
        padded_images.append(
            cv.copyMakeBorder(
                src=img,
                top=extra_border_padding,
                bottom=extra_border_padding + (extra_vertical_padding if img_idx < len(images) - 1 else 0),
                left=extra_border_padding + req_padding // 2,
                right=extra_border_padding + (req_padding - req_padding // 2),
                borderType=cv.BORDER_CONSTANT,
                value=pad_color,
            )
        )
    return cv.vconcat(padded_images)


def hconcat_and_pad_if_needed(
    images: typing.Sequence[np.ndarray],
    extra_border_padding: int = 10,
    extra_horizontal_padding: int = 20,
    pad_color: typing.Tuple[int, int, int] = default_pad_color,
) -> np.ndarray:
    """Concatenates a list of images horizontally with optional auto-padding."""
    assert all([img.ndim == 3 for img in images])
    max_height = max([img.shape[0] for img in images])
    padded_images = []
    for img_idx, img in enumerate(images):
        req_padding = max_height - img.shape[0]
        padded_images.append(
            cv.copyMakeBorder(
                src=img,
                top=extra_border_padding + req_padding // 2,
                bottom=extra_border_padding + (req_padding - req_padding // 2),
                left=extra_border_padding,
                right=extra_border_padding + (extra_horizontal_padding if img_idx < len(images) - 1 else 0),
                borderType=cv.BORDER_CONSTANT,
                value=pad_color,
            )
        )
    return cv.hconcat(padded_images)
