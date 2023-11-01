import functools
import itertools

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from hardpicks.data.fbp.auto_pick_parser import AutoPickParser
from hardpicks.data.fbp.collate import fbp_batch_collate
from hardpicks.data.fbp.constants import BAD_FIRST_BREAK_PICK_INDEX
from hardpicks.metrics.fbp.preconfigured_evaluator import get_preconfigured_regression_evaluator


@pytest.fixture
def fake_targets_and_predictions_dataframe():
    np.random.seed(23423)

    unique_shot_ids = np.random.randint(1, 1e9, 10)
    unique_rec_ids = np.random.randint(1, 1e9, 100)

    number_of_possible_picks = 10

    pick_possibilities = np.concatenate([[BAD_FIRST_BREAK_PICK_INDEX], np.arange(1, number_of_possible_picks + 1)])
    weights = np.array([10] + number_of_possible_picks * [1])
    probs = weights / np.sum(weights)

    combination = list(itertools.product(unique_shot_ids, unique_rec_ids))

    shot_ids = [c[0] for c in combination]
    rec_ids = [c[1] for c in combination]

    number_of_rows = len(combination)
    targets = np.random.choice(pick_possibilities, number_of_rows, p=probs)
    predictions = np.random.choice(pick_possibilities, number_of_rows, p=probs)

    df = pd.DataFrame(
        data=dict(
            shot_id=shot_ids, rec_id=rec_ids, target=targets, prediction=predictions
        )
    )
    return df


@pytest.fixture
def pick_file_path(
    fake_targets_and_predictions_dataframe, sample_rate_in_milliseconds, tmpdir
):

    path = tmpdir.join("fake.pic")

    header_line = "ENSEMBLE NO :         {shot_number}     {shot_id}  (shot {shot_number})          0\n"
    shot_groups = fake_targets_and_predictions_dataframe.groupby("shot_id")

    with open(path, "w") as f:
        f.write("# some comment\n")
        f.write("\n")
        f.write("\n")
        for shot_number, (shot_id, shot_df) in enumerate(shot_groups):
            f.write(header_line.format(shot_number=shot_number, shot_id=shot_id))
            f.write("   Trace       Time\n")

            counter = 0
            for _, row in shot_df.iterrows():
                if row["prediction"] == BAD_FIRST_BREAK_PICK_INDEX:
                    continue
                counter += 1
                t = sample_rate_in_milliseconds * row["prediction"]
                id = int(row["rec_id"])
                f.write(f"{counter:8d}   {t:5.3f}   {id:8d}\n")
            f.write("\n")

    return path


@pytest.fixture
def dataloader(fake_targets_and_predictions_dataframe):

    number_of_rows = len(fake_targets_and_predictions_dataframe)

    gather_size = 10

    list_data = []
    gather_id = 0
    for idx in np.arange(0, number_of_rows, gather_size):
        gather_id += 1

        sub_df = fake_targets_and_predictions_dataframe[idx: idx + gather_size]
        shot_ids = np.unique(sub_df["shot_id"].values)
        rec_ids = sub_df["rec_id"].values
        first_break_labels = sub_df["target"].values
        samples = np.random.rand(gather_size, 30)
        offset_distances = np.random.rand(gather_size, 2)

        assert len(shot_ids) == 1
        data = dict(
            shot_id=shot_ids[0],
            origin=0,
            gather_id=gather_id,
            rec_ids=rec_ids,
            first_break_labels=first_break_labels,
            offset_distances=offset_distances,
            samples=samples,
        )
        list_data.append(data)

    collate_fn = functools.partial(fbp_batch_collate, pad_to_nearest_pow2=False)
    return DataLoader(list_data, shuffle=False, batch_size=5, collate_fn=collate_fn)


@pytest.fixture()
def expected_summary(fake_targets_and_predictions_dataframe):
    targets = fake_targets_and_predictions_dataframe["target"].values
    predictions = fake_targets_and_predictions_dataframe["prediction"].values

    valid_targets_mask = targets > BAD_FIRST_BREAK_PICK_INDEX

    valid_targets = targets[valid_targets_mask]
    valid_predictions = predictions[valid_targets_mask]

    number_of_valid_targets = len(valid_targets)

    errors = valid_predictions - valid_targets

    absolute_errors = np.abs(errors)

    summary = dict()
    for delta in [1, 3, 5, 7, 9]:
        summary[f"HitRate{delta}px"] = (
            np.sum(absolute_errors < delta) / number_of_valid_targets
        )

    summary["MeanAbsoluteError"] = np.nanmean(absolute_errors)
    summary["MeanBiasError"] = np.nanmean(errors)
    summary["RootMeanSquaredError"] = np.sqrt(np.nanmean(errors ** 2))

    return summary


@pytest.mark.parametrize("sample_rate_in_milliseconds", [1, 2, 3])
def test_auto_pick_parser(
    pick_file_path, sample_rate_in_milliseconds, dataloader, expected_summary
):
    evaluator = get_preconfigured_regression_evaluator()
    auto_pick_parser = AutoPickParser(
        site_name="test",
        shot_identifier_key='shot_id',
        autopicks_file_path=pick_file_path,
        sample_rate_in_milliseconds=sample_rate_in_milliseconds,
    )

    for batch_idx, batch in enumerate(dataloader):
        raw_preds = auto_pick_parser.get_raw_preds(batch)
        evaluator.ingest(batch, batch_idx, raw_preds)
    computed_summary = evaluator.summarize()

    for key in computed_summary.keys():

        computed = computed_summary[key]
        expected = expected_summary[key]

        np.testing.assert_almost_equal(computed, expected)
