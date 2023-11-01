import torch

import hardpicks.models.fbp.utils as model_utils


def _generate_minibatch_dict(
    batch_size: int,
    trace_count: int,
    sample_count: int,
):
    return {
        # for now, we'll pretend there's a single useful field in the batch dict
        "samples": torch.randn(batch_size, trace_count, sample_count),
        # todo: add extra channel stuff that might be useful later on
    }


def test_prepare_vanilla_input_features():
    batch_size, trace_count, sample_count = 10, 64, 1024
    fake_minibatch_dict = _generate_minibatch_dict(batch_size, trace_count, sample_count)
    input_tensor = model_utils.prepare_input_features(
        fake_minibatch_dict,
        use_dist_offsets=False,
        use_first_break_prior=False,
        augmentations=None,
    )
    assert isinstance(input_tensor, torch.Tensor)
    assert input_tensor.ndim == 4
    assert input_tensor.shape[0] == batch_size
    assert input_tensor.shape[1] == 1  # we're just using the amplitude samples, so 1ch
    assert input_tensor.shape[2] == trace_count
    assert input_tensor.shape[3] == sample_count
