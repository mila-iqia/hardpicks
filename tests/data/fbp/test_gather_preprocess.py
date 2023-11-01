import numpy as np
import pytest

from hardpicks.data.fbp.gather_preprocess import ShotLineGatherPreprocessor


@pytest.fixture()
def fake_mini_samples_dataset():
    dataset = []
    trace_length = 1000  # should not really matter, as long as it's big enough
    trace_raw_amplitude = (1.0 + np.abs(np.random.randn())) * 3  # we just want a bigger-than-1 val
    for gather_idx in range(100):
        trace_count = 30 + np.random.randint(200)
        trace_samples = np.random.randn(trace_count, trace_length,) * trace_raw_amplitude
        dataset.append({"samples": trace_samples})
    return dataset


def test_gather_preprocess_default_values(fake_mini_samples_dataset):
    # start off with a simple check to make sure that constructing with default values works
    wrapped_dataset = ShotLineGatherPreprocessor(
        dataset=fake_mini_samples_dataset,
    )
    assert len(wrapped_dataset) == len(fake_mini_samples_dataset)


def test_gather_normalization_tracewise_absmax(fake_mini_samples_dataset):
    wrapped_dataset = ShotLineGatherPreprocessor(
        dataset=fake_mini_samples_dataset,
        normalize_samples=True,
        normalize_offsets=False,
        sample_norm_strategy="tracewise-abs-max",
        generate_segm_masks=False,
        augmentations=None,
    )
    assert len(wrapped_dataset) == len(fake_mini_samples_dataset)
    for gather_idx in range(len(wrapped_dataset)):
        orig_samples = fake_mini_samples_dataset[gather_idx]["samples"]
        wrapped_samples = wrapped_dataset[gather_idx]["samples"]
        assert orig_samples.shape == wrapped_samples.shape
        for orig_trace_samples, wrapped_trace_samples in \
                zip(orig_samples, wrapped_samples):
            assert wrapped_trace_samples.min() >= -1.
            assert wrapped_trace_samples.max() <= 1.
            absmax_per_trace = np.max(np.abs(orig_trace_samples))
            restored_trace_samples = wrapped_trace_samples * absmax_per_trace
            assert np.isclose(restored_trace_samples, orig_trace_samples).all()


@pytest.fixture()
def fake_mini_offsets_dataset():
    dataset = []
    for gather_idx in range(100):
        trace_count = 30 + np.random.randint(100)
        base_shot_dist = 10. + np.random.rand() * 3000
        offset_dists = []
        for rec_idx in range(trace_count):
            next_rec_dist = 5. + np.random.rand() * 50
            prev_rec_dist = 5. + np.random.rand() * 50
            offset_dists.append([base_shot_dist, next_rec_dist, prev_rec_dist])
            base_shot_dist += np.abs(np.random.randn() * 25)
        offset_dists = np.asarray(offset_dists)
        dataset.append({"offset_distances": offset_dists, "trace_count": trace_count})
    return dataset


def test_gather_normalization_offsets(fake_mini_offsets_dataset):
    wrapped_dataset = ShotLineGatherPreprocessor(
        dataset=fake_mini_offsets_dataset,
        normalize_samples=False,
        normalize_offsets=True,
    )
    assert len(wrapped_dataset) == len(fake_mini_offsets_dataset)
    for gather_idx in range(len(wrapped_dataset)):
        orig_offsets = fake_mini_offsets_dataset[gather_idx]["offset_distances"]
        wrapped_offsets = wrapped_dataset[gather_idx]["offset_distances"]
        assert orig_offsets.shape == wrapped_offsets.shape
        for orig_dists, wrapped_dists in \
                zip(orig_offsets, wrapped_offsets):
            assert wrapped_dists.min() > 0. and wrapped_dists.max() < 3.


@pytest.fixture()
def fake_mini_samples_and_fb_labels_dataset():
    dataset = []
    trace_length = 500  # should not really matter, as long as it's big enough
    trace_raw_amplitude = (1.0 + np.abs(np.random.randn())) * 3  # we just want a bigger-than-1 val
    for gather_idx in range(100):
        trace_count = 30 + np.random.randint(200)
        trace_samples = np.random.randn(trace_count, trace_length, ) * trace_raw_amplitude
        first_break_labels = 1 + np.random.randint(trace_length - 1, size=(trace_count, ))
        bad_trace_count = min(trace_count, np.random.randint(100))
        secret_bad_traces = np.random.choice(np.arange(trace_count), bad_trace_count)
        first_break_labels[secret_bad_traces] = -1
        dataset.append({
            "samples": trace_samples,
            "first_break_labels": first_break_labels,
            "secret_bad_traces": secret_bad_traces,
        })
    return dataset


@pytest.mark.slow
@pytest.mark.parametrize("segm_class_count", [1, 2, 3])
def test_gather_mask_generation(fake_mini_samples_and_fb_labels_dataset, segm_class_count):
    wrapped_dataset = ShotLineGatherPreprocessor(
        dataset=fake_mini_samples_and_fb_labels_dataset,
        normalize_samples=False,
        normalize_offsets=False,
        generate_segm_masks=True,
        segm_class_count=segm_class_count,
        augmentations=None,
    )
    assert len(wrapped_dataset) == len(fake_mini_samples_and_fb_labels_dataset)
    for gather_idx in range(len(wrapped_dataset)):
        orig_samples = fake_mini_samples_and_fb_labels_dataset[gather_idx]["samples"]
        first_break_labels = fake_mini_samples_and_fb_labels_dataset[gather_idx]["first_break_labels"]
        assert np.array_equal(first_break_labels, wrapped_dataset[gather_idx]["first_break_labels"])
        segm_mask = wrapped_dataset[gather_idx]["segmentation_mask"]
        assert segm_mask.shape == orig_samples.shape
        secret_bad_traces = fake_mini_samples_and_fb_labels_dataset[gather_idx]["secret_bad_traces"]
        for trace_idx in range(len(segm_mask)):
            if trace_idx in secret_bad_traces:
                assert (segm_mask[trace_idx] == -1).all()
            else:
                first_break_label = first_break_labels[trace_idx]
                if segm_class_count == 1:
                    for col_idx in range(0, first_break_label):
                        assert segm_mask[trace_idx, col_idx] == 0
                    assert segm_mask[trace_idx, first_break_label] == 1
                    for col_idx in range(first_break_label + 1, segm_mask.shape[1]):
                        assert segm_mask[trace_idx, col_idx] == 0
                elif segm_class_count == 2:
                    for col_idx in range(0, first_break_label):
                        assert segm_mask[trace_idx, col_idx] == 0
                    for col_idx in range(first_break_label, segm_mask.shape[1]):
                        assert segm_mask[trace_idx, col_idx] == 1
                elif segm_class_count == 3:
                    for col_idx in range(0, first_break_label):
                        assert segm_mask[trace_idx, col_idx] == 0
                    assert segm_mask[trace_idx, first_break_label] == 2
                    for col_idx in range(first_break_label + 1, segm_mask.shape[1]):
                        assert segm_mask[trace_idx, col_idx] == 1
