import numpy as np
import pytest
import sklearn.metrics
import torch

import hardpicks.metrics.fbp.utils as utils


def test_get_interval_from_break_points():
    break_points = [1, 32, 1000, 1e6]
    ret = utils.get_intervals_from_break_points(break_points)
    assert len(ret) == len(break_points) + 1
    last_interval_max = 0.0
    for interval_idx, interval in enumerate(ret):
        if interval_idx >= len(break_points):
            expected_interval_max = float("inf")
        else:
            expected_interval_max = break_points[interval_idx]
        assert interval[1] == expected_interval_max
        assert interval[0] == last_interval_max
        last_interval_max = interval[1]


def test_assign_interval_ids():
    for _ in range(1000):
        interval_buckets, sample_bucket_ids, samples = [], [], []
        bucket_count = 1 + np.random.randint(10)
        latest_interval_max = 0.0
        for bucket_idx in range(bucket_count):
            new_interval_max = latest_interval_max + 1.0 + np.random.random() * 10
            interval_buckets.append((latest_interval_max, new_interval_max))
            curr_bucket_sample_count = np.random.randint(100)
            curr_bucket_range = new_interval_max - latest_interval_max
            for _ in range(curr_bucket_sample_count):
                samples.append(
                    latest_interval_max + np.random.random() * curr_bucket_range
                )
                sample_bucket_ids.append(bucket_idx)
            latest_interval_max = new_interval_max
        shuffled_idxs = np.random.permutation(len(samples))
        samples = np.asarray(samples)[shuffled_idxs]
        sample_bucket_ids = np.asarray(sample_bucket_ids)[shuffled_idxs]
        ret = utils.assign_interval_ids(np.asarray(samples), interval_buckets)
        assert np.array_equal(ret, sample_bucket_ids)


def test_get_regr_preds_from_raw_preds_with_segm_class_count_is_none():
    # first, test that the input preds are returned as-is if we're not doing segmentation
    ret, prob = utils.get_regr_preds_from_raw_preds(
        raw_preds="potato", segm_class_count=None
    )
    assert prob is None
    assert ret == "potato"


def get_gaussian_probability_array(list_times, mu, sigma, maximum_probability):
    list_z2 = (list_times - mu) ** 2 / (2.0 * sigma**2)
    return maximum_probability * np.exp(-list_z2)


def get_sigmoid_probability_array(list_times, mu, sigma, maximum_probability):
    list_z = (list_times - mu) / sigma
    return maximum_probability / (1 + np.exp(-list_z))


def get_index_of_maximum_above_threshold(array, threshold):
    index = np.argmax(array)
    if array[index] >= threshold:
        return index
    else:
        return 0


def get_index_of_first_above_threshold(array, threshold):
    indices_above_threshold = np.where(array >= threshold)[0]
    if len(indices_above_threshold) == 0:
        return 0
    else:
        return indices_above_threshold[0]


def get_prediction_map_and_fist_break_indices(
    nclasses,
    maximum_probability,
    probability_threshold,
    probability_function,
    index_function,
):
    batch_size = 4
    number_of_traces = 12
    number_of_samples = 117

    list_t = np.linspace(0, 1, number_of_samples)

    list_sigma = np.random.rand(batch_size, number_of_traces)
    list_mu = np.random.rand(batch_size, number_of_traces)

    # create a random prediction map with dimensions [batch size, classes, traces, samples]

    # instantiate array with dummy data
    first_break_indices = -np.ones([batch_size, number_of_traces]).astype(np.int32)
    probability_map = -np.ones(
        [batch_size, nclasses, number_of_traces, number_of_samples]
    )

    for batch_idx in range(batch_size):
        for trace_idx in range(number_of_traces):
            sigma = list_sigma[batch_idx, trace_idx]
            mu = list_mu[batch_idx, trace_idx]

            # create a fbp probability over time centered at mu
            first_break_probabilities = probability_function(
                list_t, mu, sigma, maximum_probability
            )
            first_break_indices[batch_idx, trace_idx] = index_function(
                first_break_probabilities, probability_threshold
            )

            # the other class probabilities will be randomly chosen so that the total
            # probability at a given time sums to one.

            left_over_probs = 1.0 - first_break_probabilities
            for class_idx in range(nclasses - 2):
                class_probs = left_over_probs * np.random.rand(number_of_samples)
                probability_map[batch_idx, class_idx, trace_idx, :] = class_probs
                left_over_probs = left_over_probs - class_probs
            probability_map[batch_idx, nclasses - 2, trace_idx, :] = left_over_probs
            probability_map[
                batch_idx, nclasses - 1, trace_idx, :
            ] = first_break_probabilities

    # convert probabilities to logits, as "scores" are expected downsteam
    constants = np.random.rand(batch_size, number_of_traces, number_of_samples)
    constants = np.repeat(constants[:, np.newaxis, :, :], nclasses, axis=1)
    logit_maps = np.log(probability_map) + constants

    return logit_maps, first_break_indices


@pytest.fixture()
def prediction_map_and_fist_break_indices(
    segm_class_count, max_fbp_prob, prob_threshold
):
    np.random.seed(23423)

    if segm_class_count == 1:
        nclasses = 2
    else:
        nclasses = segm_class_count

    if segm_class_count == 1 or segm_class_count == 3:
        probability_function = get_gaussian_probability_array
        index_function = get_index_of_maximum_above_threshold
    else:
        assert segm_class_count == 2
        probability_function = get_sigmoid_probability_array
        index_function = get_index_of_first_above_threshold

    logit_maps, first_break_indices = get_prediction_map_and_fist_break_indices(
        nclasses, max_fbp_prob, prob_threshold, probability_function, index_function
    )

    first_break_probabilities = utils.get_first_break_class_prob_map(
        torch.Tensor(logit_maps), segm_class_count
    )

    batch_size, number_of_traces, _ = first_break_probabilities.shape
    probabilities_of_fbp = np.zeros([batch_size, number_of_traces])

    for bs in range(batch_size):
        for trace_idx in range(number_of_traces):
            sample_idx = first_break_indices[bs, trace_idx]
            p = first_break_probabilities[bs, trace_idx, sample_idx]
            if p < prob_threshold:
                p = np.NaN
            probabilities_of_fbp[bs, trace_idx] = p

    return (
        torch.tensor(logit_maps),
        torch.tensor(first_break_indices).to(torch.long),
        torch.tensor(probabilities_of_fbp),
    )


@pytest.mark.parametrize(
    "segm_class_count,  max_fbp_prob, prob_threshold",
    [
        (1, 1.0, 0.5),
        (1, 0.25, 0.5),
        (2, 1.0, 0.5),
        (2, 0.25, 0.5),
        (3, 1.0, 0.75),
        (3, 0.35, 0.75),
    ],
)
def test_get_regr_preds_from_raw_preds(
    segm_class_count,
    prediction_map_and_fist_break_indices,
    max_fbp_prob,
    prob_threshold,
):

    (
        raw_prediction_map,
        expected_first_break_indices,
        expected_probabilities_of_fbp,
    ) = prediction_map_and_fist_break_indices

    (
        computed_first_break_indices,
        computed_probabilities_of_fbp,
    ) = utils.get_regr_preds_from_raw_preds(
        raw_prediction_map,
        segm_class_count,
        prob_threshold=prob_threshold,
    )

    torch.testing.assert_allclose(
        expected_probabilities_of_fbp, computed_probabilities_of_fbp
    )

    if prob_threshold <= max_fbp_prob:
        torch.testing.assert_allclose(
            computed_first_break_indices, expected_first_break_indices
        )
    else:
        zeros = torch.zeros_like(computed_first_break_indices)
        torch.testing.assert_allclose(computed_first_break_indices, zeros)


def test_get_probabilities_of_fbp_indices():

    np.random.seed(34523)
    batch_size = 4
    number_of_traces = 8
    number_of_samples = 41

    threshold = 0.6
    fbp_probabilities = np.zeros([batch_size, number_of_traces, number_of_samples])
    fbp_indices = np.zeros([batch_size, number_of_traces], dtype=int)
    expected_probabilities = np.zeros([batch_size, number_of_traces])

    for bs in range(batch_size):
        for trace_idx in range(number_of_traces):
            probs = threshold * np.random.random(number_of_samples)

            # flip a coin to see if pick is bad
            if np.random.rand() > 0.5:
                idx = 0
                p = np.NaN
            else:
                idx = np.random.randint(1, number_of_samples)
                p = threshold + (1.0 - threshold) * np.random.rand()
                probs[idx] = p

            fbp_probabilities[bs, trace_idx] = probs
            fbp_indices[bs, trace_idx] = idx
            expected_probabilities[bs, trace_idx] = p

    fbp_probabilities = torch.Tensor(fbp_probabilities)
    fbp_indices = torch.Tensor(fbp_indices).to(torch.long)
    expected_probabilities = torch.Tensor(expected_probabilities)

    computed_probabilities = utils._get_probabilities_of_fbp_indices(
        fbp_probabilities, fbp_indices
    )

    torch.testing.assert_allclose(expected_probabilities, computed_probabilities)


def test_compute_iou_hardcoded():
    assert np.isnan(utils.compute_iou_from_classif_counts(0, 0, 0))
    assert (
        utils.compute_iou_from_classif_counts(
            true_positives=1,
            false_positives=0,
            false_negatives=0,
        )
        == 1.0
    )
    assert (
        utils.compute_iou_from_classif_counts(
            true_positives=0,
            false_positives=0,
            false_negatives=1,
        )
        == 0.0
    )
    assert (
        utils.compute_iou_from_classif_counts(
            true_positives=0,
            false_positives=1,
            false_negatives=0,
        )
        == 0.0
    )
    assert (
        utils.compute_iou_from_classif_counts(
            true_positives=0,
            false_positives=1,
            false_negatives=1,
        )
        == 0.0
    )
    assert (
        utils.compute_iou_from_classif_counts(
            true_positives=1,
            false_positives=1,
            false_negatives=1,
        )
        == 1 / 3
    )


def test_compute_iou_vs_sklearn():
    map_size = (1000,)
    for _ in range(1000):
        y_true = np.random.randint(2, size=map_size)
        y_pred = np.random.randint(2, size=map_size)
        # set hot corner to avoid nan/warnings/throwing
        y_true[0], y_pred[0] = 1, 1
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        iou_homemade = utils.compute_iou_from_classif_counts(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
        assert not np.isnan(iou_homemade)
        assert 0 < iou_homemade <= 1.0
        iou_sklearn = sklearn.metrics.jaccard_score(y_true, y_pred)
        assert iou_homemade == iou_sklearn
