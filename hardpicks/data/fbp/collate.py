import typing

import numpy as np
import torch.utils.data
import torch.utils.data._utils

import hardpicks.utils.processing_utils as processing_utils


def get_fields_to_pad() -> typing.Sequence:
    """Returns the fields to be padded across all gather parser/cleaner/preprocessor classes."""
    # note: this is not a constant anymore to avoid circular import issues!
    import hardpicks.data.fbp.gather_cleaner as gather_cleaner
    import hardpicks.data.fbp.gather_parser as gather_parser
    import hardpicks.data.fbp.gather_preprocess as gather_preprocess
    return [
        *gather_parser.ShotLineGatherDataset.variable_length_fields,
        *gather_cleaner.ShotLineGatherCleaner.variable_length_fields,
        *gather_preprocess.ShotLineGatherPreprocessor.variable_length_fields,
    ]


def get_fields_to_double_pad() -> typing.Sequence:
    """Returns the fields to be double-axis-padded all gather parser/cleaner/preprocessor classes."""
    # we cheat a little bit here and specify the names that need to be double-padded directly
    return ["samples", "segmentation_mask", "first_break_prior"]


def fbp_batch_collate(batch, pad_to_nearest_pow2):
    """Puts each data field into a tensor with outer dimension batch size."""
    # first, we'll do a global loop to validate the max array size
    max_len = _get_max_first_dimension(batch)
    max_samples = _get_max_second_dimension(batch)

    if pad_to_nearest_pow2:
        max_len = processing_utils.get_nearest_power_of_two(max_len)
        max_samples = processing_utils.get_nearest_power_of_two(max_samples)

    # now, do a 2nd loop where we do the actual padding (per-instance)
    # TODO: if collate is found to be a bottleneck later, we could revisit this section to pin+pad
    #       ... or put the 2nd axis padding in gather preprocessing, and pin+pad only 1st axis here
    fields_to_pad, fields_to_double_pad = get_fields_to_pad(), get_fields_to_double_pad()
    for sample in batch:
        for field_to_pad, pad_value in fields_to_pad:
            if field_to_pad not in sample or sample[field_to_pad] is None:
                continue
            if field_to_pad in fields_to_double_pad:
                assert sample[field_to_pad].ndim == 2
                sample[field_to_pad] = np.pad(
                    array=sample[field_to_pad],
                    pad_width=[
                        (0, max_len - sample[field_to_pad].shape[0]),
                        (0, max_samples - sample[field_to_pad].shape[1]),
                    ],
                    mode="constant",
                    constant_values=pad_value,
                )
            else:
                sample[field_to_pad] = np.pad(
                    array=sample[field_to_pad],
                    pad_width=[
                        (0, max_len - len(sample[field_to_pad])),
                        *[(0, 0) for _ in range(1, len(sample[field_to_pad].shape))]
                    ],
                    mode="constant",
                    constant_values=pad_value,
                )
    # now, just forward everything to the default collate to actually do the stacking (+pinning)
    output = torch.utils.data._utils.collate.default_collate(batch)
    assert "batch_size" not in output, "the 'batch_size' keyword in batch dictionaries is RESERVED!"
    output["batch_size"] = len(batch)
    return output


def _get_max_first_dimension(batch):
    dim = 0
    fields_to_pad = [field_to_pad for field_to_pad, _ in get_fields_to_pad()]
    return _get_max_dimension(batch, fields_to_pad, dim)


def _get_max_second_dimension(batch):
    dim = 1
    return _get_max_dimension(batch, get_fields_to_double_pad(), dim)


def _get_max_dimension(batch, fields_to_pad, dim):
    max_dimension = None
    for field_to_pad in fields_to_pad:
        curr_max_dimension = 0
        for sample in batch:
            if field_to_pad not in sample or sample[field_to_pad] is None:
                continue
            assert isinstance(sample[field_to_pad], np.ndarray)
            assert len(sample[field_to_pad].shape) >= 1
            curr_max_dimension = max(curr_max_dimension, sample[field_to_pad].shape[dim])
        if curr_max_dimension == 0:
            continue
        assert max_dimension is None or max_dimension == curr_max_dimension, "unexpected width mismatch across fields"
        max_dimension = curr_max_dimension
    assert max_dimension is not None
    return max_dimension
