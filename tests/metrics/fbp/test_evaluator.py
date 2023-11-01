import os
import typing

import deepdiff
import numpy as np
import pytest
import torch

import hardpicks.metrics.fbp.evaluator as eval
from hardpicks.data.fbp.constants import (
    BAD_FIRST_BREAK_PICK_INDEX,
    BAD_OR_PADDED_ELEMENT_ID,
    SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP,
)
from hardpicks.models.constants import DONTCARE_SEGM_MASK_LABEL


@pytest.fixture
def segm_eval_metrics():
    return [
        {"metric_type": "GatherCoverage"},
    ]


@pytest.fixture
def regr_eval_metrics():
    return [
        {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 1}},
        {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 3}},
        {"metric_type": "MeanAbsoluteError"},
        {"metric_type": "RootMeanSquaredError"},
        {"metric_type": "MeanBiasError"},
    ]


@pytest.fixture
def all_eval_metrics(segm_eval_metrics, regr_eval_metrics):
    # the config below is the same as the one in the 'unet-mini' test config
    return [*segm_eval_metrics, *regr_eval_metrics]


def _get_evaluator(
    eval_metrics: typing.Dict[typing.AnyStr, typing.Dict[typing.AnyStr, typing.Any]],
    segm_class_count: int,
):
    evaluator = eval.FBPEvaluator(
        hyper_params={
            "eval_metrics": eval_metrics,
            "segm_class_count": segm_class_count,
            "segm_first_break_prob_threshold": 0.5,
        }
    )
    # and check the attributes we just passed in
    assert len(evaluator.metrics_metamap) == len(eval_metrics)
    assert evaluator.segm_class_count == segm_class_count
    return evaluator


def test_regr_only_metrics_initialization(regr_eval_metrics):
    # if we don't ask for segm metrics, they should not be required and used
    evaluator = _get_evaluator(regr_eval_metrics, segm_class_count=None)
    assert len(evaluator.metrics_metamap) == len(regr_eval_metrics)


def test_segm_only_metrics_initialization(segm_eval_metrics):
    # if we ask for segm metrics, we need to provide a class count
    with pytest.raises(AssertionError):
        _ = _get_evaluator(segm_eval_metrics, segm_class_count=None)
    # ... and the metrics should now be required and used
    evaluator = _get_evaluator(segm_eval_metrics, segm_class_count=2)
    assert len(evaluator.metrics_metamap) == len(segm_eval_metrics)


def test_all_metrics_initialization(all_eval_metrics):
    # here, we'll check that segm+regr metrics can be combo'd and that counter names are all good
    for class_count in SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP:
        evaluator = _get_evaluator(all_eval_metrics, segm_class_count=class_count)
        assert len(evaluator.metrics_metamap) == len(all_eval_metrics)


def test_segm_coverage_array(all_eval_metrics):
    # here, a 'false' flag should be assigned for each trace which does not get a good FB prediction
    for class_count in SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP:
        evaluator = _get_evaluator(all_eval_metrics, segm_class_count=class_count)
        for _ in range(1000):
            ccount = len(SEGM_CLASS_COUNT_TO_CLASS_NAMES_MAP[class_count])
            sample_count = 100
            nb_traces = 25
            pred_first_break_labels = np.random.randint(
                10, sample_count, size=(nb_traces,)
            )
            bad_pred_count = np.random.randint(nb_traces)
            bad_preds = np.random.permutation(nb_traces)[:bad_pred_count]
            pred_first_break_labels[bad_preds] = BAD_FIRST_BREAK_PICK_INDEX
            target_class_map = np.random.randint(ccount, size=(nb_traces, sample_count))
            invalid_idxs_count = np.random.randint(nb_traces)
            invalid_idxs = np.random.permutation(nb_traces)[:invalid_idxs_count]
            target_class_map[invalid_idxs, :] = BAD_OR_PADDED_ELEMENT_ID
            good_flags, expected_flags = evaluator._get_segm_coverage_array(
                target_class_map,
                pred_first_break_labels,
            )
            all_removed_idxs = np.unique([*bad_preds, *invalid_idxs])
            assert good_flags.sum() == nb_traces - len(all_removed_idxs)
            assert expected_flags.sum() == nb_traces - invalid_idxs_count


def _generate_fake_batch_data(
    ccount: int,
    shot_id: int = 0,
):
    # this data is fake and totally random, we don't actually use it to check metric results...
    sample_count, trace_count, gather_count = 100, 25, 10
    batch = {
        "samples": torch.Tensor(
            np.random.randn(gather_count, trace_count, sample_count)
        ),
        "segmentation_mask": torch.Tensor(
            np.random.randint(ccount, size=(gather_count, trace_count, sample_count))
        ).long(),
        "first_break_labels": torch.Tensor(
            10 + np.random.randint(sample_count - 10, size=(gather_count, trace_count))
        ).long(),  # note: this fake batch contains no trace with an invalid FB label
        "offset_distances": torch.Tensor(np.random.randn(gather_count, trace_count, 1)),
        "origin": ["potato"] * gather_count,
        "rec_ids": torch.Tensor(
            np.tile(np.arange(trace_count), gather_count).reshape(
                gather_count, trace_count
            )
        ).long(),
        "shot_id": [shot_id] * gather_count,
        "gather_id": (np.arange(gather_count) + gather_count * (shot_id + 1)).tolist(),
        "batch_size": gather_count,
    }

    return batch, gather_count, trace_count, sample_count


def test_metrics_arrays(all_eval_metrics):
    # we'll just make a dummy batch and test the shape of the output arrays
    ccount = 3  # class count doesn't really matter here anymore, we'll just test I/O
    evaluator = _get_evaluator(all_eval_metrics, segm_class_count=ccount)
    batch, gather_count, trace_count, sample_count = _generate_fake_batch_data(ccount)
    pred_first_breaks = torch.Tensor(
        10 + np.random.randint(sample_count - 10, size=(gather_count, trace_count))
    )
    good_traces_per_gather = [
        np.arange(trace_count)[np.random.permutation(np.random.randint(5, 15))]
        for _ in range(gather_count)
    ]
    segm_results = evaluator._get_segm_metrics_arrays(
        batch,
        pred_first_breaks.long(),
        good_traces_per_gather,
    )
    good_traces_total = sum([len(idxs) for idxs in good_traces_per_gather])
    assert "GatherCoverage" in segm_results
    assert all([len(array) == good_traces_total for array in segm_results.values()])
    regr_results = evaluator._get_regr_error_array(
        batch,
        pred_first_breaks,
        good_traces_per_gather,
    )
    assert len(regr_results) == good_traces_total


def test_ingest_and_summarize(all_eval_metrics):
    # here, we'll geberate metrics for two random batches...
    evaluator = _get_evaluator(all_eval_metrics, segm_class_count=3)

    batch1, gather_count1, trace_count1, sample_count1 = _generate_fake_batch_data(
        evaluator.segm_class_count, 1
    )
    batch2, gather_count2, trace_count2, sample_count2 = _generate_fake_batch_data(
        evaluator.segm_class_count, 2
    )

    # make sure the evaluation results are not the same for both (although keys must overlap...)
    pred_scores_maps1 = torch.Tensor(
        np.random.randn(
            gather_count1, evaluator.segm_class_count, trace_count1, sample_count1
        )
    )
    metrics1 = evaluator.ingest(batch1, 1, pred_scores_maps1)
    evaluator.finalize()
    assert len(evaluator._dataframe) == gather_count1 * trace_count1
    pred_scores_maps2 = torch.Tensor(
        np.random.randn(
            gather_count2, evaluator.segm_class_count, trace_count2, sample_count2
        )
    )
    metrics2 = evaluator.ingest(batch2, 2, pred_scores_maps2)
    evaluator.finalize()
    assert (
        len(evaluator._dataframe)
        == gather_count2 * trace_count2 + gather_count1 * trace_count1
    )
    assert len(np.intersect1d(list(metrics1.keys()), list(metrics2.keys()))) == len(
        metrics1
    )
    assert not all([v1 == v2 for v1, v2 in zip(metrics1.values(), metrics2.values())])

    # now, just get the summarized version of the internal dataframe
    metrics_summarized = evaluator.summarize()
    # ... the results should also differ from the last output (which was a partial summarization)
    assert len(
        np.intersect1d(list(metrics_summarized.keys()), list(metrics2.keys()))
    ) == len(metrics1)
    assert not all(
        [v1 == v2 for v1, v2 in zip(metrics_summarized.values(), metrics2.values())]
    )


def test_ingest_without_labels(all_eval_metrics):
    """
    The goal of this test is to make sure the code fails hard if a gather's segmentation mask is full
    of "don't care" labels.
    """
    np.random.seed(213423)

    evaluator = _get_evaluator(all_eval_metrics, segm_class_count=3)
    # these parameters don't matter
    shot_id = 37
    batch_idx = 42

    batch, gather_count, trace_count, sample_count = _generate_fake_batch_data(
        evaluator.segm_class_count, shot_id
    )

    # Fill the segmentation mask of one gather with non-data to make sure the code fails hard!
    bad_data_gather_idx = np.random.randint(gather_count)
    batch["segmentation_mask"][bad_data_gather_idx, :, :] = DONTCARE_SEGM_MASK_LABEL
    pred_scores_maps = torch.Tensor(
        np.random.randn(
            gather_count, evaluator.segm_class_count, trace_count, sample_count
        )
    )
    with pytest.raises(AssertionError):
        _ = evaluator.ingest(batch, batch_idx, pred_scores_maps)


def test_eval_dump_and_load(all_eval_metrics, tmpdir):
    # test that dumping, resetting, and loading returns the proper state
    evaluator = _get_evaluator(all_eval_metrics, segm_class_count=3)
    for idx, shot_id in enumerate([1, 2, 3]):
        batch, gather_count, trace_count, sample_count = _generate_fake_batch_data(
            evaluator.segm_class_count, 1
        )
        pred_scores_maps = torch.Tensor(
            np.random.randn(
                gather_count, evaluator.segm_class_count, trace_count, sample_count
            )
        )
        evaluator.ingest(batch, idx, pred_scores_maps)

    before_accumulated_trace_counts = evaluator.accumulated_trace_counts

    first_pass_res = evaluator.summarize()
    dump_path = os.path.join(tmpdir, "dummy.pkl")
    evaluator.dump(dump_path)
    evaluator.reset()

    assert len(evaluator.list_batch_dataframes) == 0
    assert evaluator.accumulated_trace_counts == 0

    reset_res = evaluator.summarize()
    assert not reset_res
    evaluator = evaluator.load(dump_path)
    after_accumulated_trace_counts = evaluator.accumulated_trace_counts

    assert after_accumulated_trace_counts == before_accumulated_trace_counts

    second_pass_res = evaluator.summarize()
    assert not deepdiff.DeepDiff(first_pass_res, second_pass_res)


@pytest.mark.parametrize("segm_class_count", [1, 2, 3])
def test_ingested_fbp_probabilities(all_eval_metrics, segm_class_count):
    evaluator = _get_evaluator(all_eval_metrics, segm_class_count=segm_class_count)

    if segm_class_count == 1:
        number_of_classes = 2
    else:
        number_of_classes = segm_class_count

    list_batches = []
    list_raw_preds = []
    for batch_idx, shot_id in enumerate([1, 2]):
        batch, gather_count, trace_count, sample_count = _generate_fake_batch_data(
            evaluator.segm_class_count, shot_id
        )
        raw_preds = torch.Tensor(
            np.random.randn(gather_count, number_of_classes, trace_count, sample_count)
        )
        list_raw_preds.append(raw_preds)
        list_batches.append(batch)
        evaluator.ingest(batch, batch_idx, raw_preds)

    evaluator.finalize()

    # Go through the dataframe row by row, making sure the probability is correct.
    index_columns = ["GatherId", "ShotId", "ReceiverId", "OriginId"]
    ref_df = evaluator._dataframe.set_index(index_columns)[
        ["Predictions", "Probabilities"]
    ]

    counter = 0
    for batch, raw_preds in zip(list_batches, list_raw_preds):
        for origin, rec_ids, shot_id, gather_id in zip(
            batch["origin"],
            batch["rec_ids"].cpu().numpy(),
            batch["shot_id"],
            batch["gather_id"],
        ):
            origin_id = evaluator.origin_id_map[origin]

            raw_preds_batch_idx = batch["gather_id"].index(gather_id)

            for rec_id in rec_ids:
                counter += 1
                raw_preds_trace_idx = list(rec_ids).index(rec_id)

                index = (gather_id, shot_id, rec_id, origin_id)
                row = ref_df.loc[index]

                fbp_idx = row["Predictions"]
                computed_fbp_prob = row["Probabilities"]

                if fbp_idx == 0:
                    assert np.isnan(computed_fbp_prob)
                else:
                    raw_values = raw_preds[
                        raw_preds_batch_idx, :, raw_preds_trace_idx, fbp_idx
                    ]
                    expected_fbp_prob = torch.softmax(raw_values, dim=0)[-1].item()
                    np.testing.assert_almost_equal(expected_fbp_prob, computed_fbp_prob)

    assert counter == len(evaluator._dataframe)
