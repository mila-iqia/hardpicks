#!/usr/bin/env python
#
#   INVALID GATHER VIEWER AND TAGGING SCRIPT v2
#
#     To run it, execute
#
#       python dataframe_analyser.py --config [path to yaml config file]
#
#     where the yaml config file is of the form:
#
#       local_data_directory: [path to data]
#       path_to_model_result_data: [path to model data dump]
#       output_yaml_path: [path to output]
#       starting_index: [integer]
#       image_width: [integer]
#       height_magnification: [integer]
#
#     An example of such a config file is available at:
#
#       <REPO_ROOT>/data/fbp/bad_gathers/analyser.yaml
#
#     The data loader specified in the provided config will be loaded, the evaluator dataframe
#     will be used to sort the data loader's gathers by prediction quality, and the interface
#     will display all gathers one by one.
#
#     If bad gathers were already identified in a previous session, these will be reloaded and
#     all of them will be carried over after the script is stopped. To stop the annotation
#     process, hit 'q' or 'escape'.
#
#     The color of the text in the displayed interface defines whether a gather is VALID (green)
#     or INVALID (red). Hitting 'enter' while the gather is displayed will toggle its validity and
#     move to the next gather. Hitting the space bar will toggle the display of groundtruth
#     labels on the current gather; this might be useful if the first break is hard to see under
#     the drawn groundtruth points.
#
#     To move to the previous or next gather without changing its validity label, use 'a' or 'd'
#     respectively. If your local window manager is Qt-based, you can also use left/right arrows.
#
import argparse
import logging
import os
import sys

import cv2 as cv
import numpy as np
import yaml

import hardpicks.data.data_loader as data_loader
import hardpicks.data.fbp.collate as collate
import hardpicks.metrics.fbp.evaluator as eval
import hardpicks.models.fbp.utils as model_utils
import hardpicks.utils.draw_utils as draw_utils
from hardpicks.data.fbp.gather_preprocess import (
    ShotLineGatherPreprocessor,
)


def get_configuration():
    """Prepares and returns the argument parser for the CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Config file to parse (in yaml format).", required=True
    )
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        parameters = yaml.load(stream, Loader=yaml.FullLoader)

    expected_variables = [
        "local_data_directory",
        "path_to_model_result_data",
        "output_yaml_path",
        "starting_index",
        "image_width",
        "height_magnification",
    ]

    for variable in expected_variables:
        assert (
            variable in parameters
        ), f"ERROR: the variable {variable} is not in the config file"

    # the 'local data directory' is the root folder where the HDF5 data subfolder can be found
    local_data_dir = parameters["local_data_directory"]
    assert os.path.isdir(local_data_dir), f"ERROR: {local_data_dir} does not exist!"

    # the dataframe path points to the pickled dump of the evaluator output coming from an experiment
    dataframe_path = parameters["path_to_model_result_data"]
    assert os.path.isfile(dataframe_path), f"ERROR: {dataframe_path} does not exist!"

    # the output yaml path is where the bad gathers that are flagged by the user will be saved
    output_yaml_path = parameters["output_yaml_path"]

    # the starting index is where the visualization should start
    starting_index = parameters["starting_index"]
    assert type(starting_index) is int, "ERROR: the starting_index should be an integer"
    assert (
        starting_index >= 0
    ), "ERROR: the starting_index should be greater or equal to zero"

    image_width = parameters["image_width"]
    assert type(image_width) is int, "ERROR: the image_width should be an integer"
    assert image_width >= 0, "ERROR: the image_width should be greater or equal to zero"

    height_magnification = parameters["height_magnification"]
    assert (
        type(height_magnification) is int
    ), "ERROR: the height_magnification  should be an integer"
    assert (
        height_magnification >= 0
    ), "ERROR: the height_magnification should be greater or equal to zero"

    return (
        local_data_dir,
        dataframe_path,
        output_yaml_path,
        starting_index,
        image_width,
        height_magnification,
    )


hyper_params = {
    # nothing in here actually needs to be changed!
    "segm_class_count": 1,
    "module_type": "FBPDataModule",
    "train_batch_size": 12,
    "eval_batch_size": 50,
    "num_workers": 0,
    "pin_memory": False,
    "convert_to_fp16": True,
    "convert_to_int16": True,
    "preload_trace_data": False,
    "cache_trace_metadata": True,
    "provide_offset_dists": True,
    "pad_to_nearest_pow2": False,
    "use_batch_sampler": False,
    "use_rebalancing_sampler": False,
    "skip_setup_overlap_check": True,
    "train_loader_params": [
        {"site_name": "Lalor"},
        {"site_name": "Halfmile"},
        # {"site_name": "Matagami"},
        {"site_name": "Sudbury"},
        {"site_name": "Brunswick"},
    ],
    "valid_loader_params": [],
    "test_loader_params": [],
}
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


if __name__ == "__main__":

    (
        local_data_dir,
        dataframe_path,
        output_yaml_path,
        starting_index,
        image_width,
        height_magnification,
    ) = get_configuration()

    data_module = data_loader.create_data_module(local_data_dir, hyper_params)
    data_module.prepare_data()
    data_module.setup()
    data_loader = data_module.train_dataloader()
    datasets = [d.dataset.dataset for d in data_loader.dataset.datasets]

    # next, reload the evaluator state to get access to gather-wise errors
    evaluator = eval.FBPEvaluator.load(dataframe_path)
    origin_id_map = {val: key for key, val in evaluator.origin_id_map.items()}
    datasets_map = {
        site_name: next(d for d in datasets if site_name in str(d.hdf5_path))
        for site_name in origin_id_map.values()
    }
    groupby_agg_map = {
        "ShotId": "first",
        "ReceiverId": "first",
        "OriginId": "first",
        "Offset": np.nanmean,
        "Errors": lambda x: np.nanmean(np.abs(x)),
    }
    full_df = evaluator._dataframe[[key for key in [*groupby_agg_map, "GatherId"]]]
    gather_df = full_df.groupby(["GatherId"]).agg(groupby_agg_map).reset_index()
    gather_df = gather_df.sort_values(["Errors"], ascending=False)

    # then, initialize the maps/structures we'll be storing our assessment results in
    bad_gather_map = {origin: {} for origin in datasets_map}
    if os.path.isfile(output_yaml_path):
        with open(output_yaml_path, "r") as fd:
            bad_gather_map = yaml.load(fd, Loader=yaml.FullLoader)
    for origin_name in origin_id_map.values():
        if origin_name not in bad_gather_map:
            bad_gather_map[origin_name] = {}
    curr_df_row_idx, latest_df_row_idx, curr_steps, draw_gt, normalize = (
        starting_index,
        -1,
        0,
        True,
        False,
    )
    gather_id, origin_id, origin_name, gather = None, None, None, None
    gather_is_valid, gather_batch, row = None, None, None

    # finally, run the gather display loop
    while True:
        curr_steps += 1
        if latest_df_row_idx != curr_df_row_idx:
            row = gather_df.iloc[curr_df_row_idx]
            gather_id, origin_id = int(row["GatherId"]), int(row["OriginId"])
            origin_name = origin_id_map[origin_id]
            gather = datasets_map[origin_name][gather_id]
            normalized_samples = ShotLineGatherPreprocessor.normalize_sample_with_tracewise_abs_max_strategy(
                gather["samples"]
            )
            gather["samples"] = normalized_samples

            gather_is_valid = gather_id not in bad_gather_map[origin_name]
            gather_batch = collate.fbp_batch_collate(
                [gather], pad_to_nearest_pow2=hyper_params["pad_to_nearest_pow2"],
            )
            latest_df_row_idx = curr_df_row_idx
        gather_image = model_utils.generate_pred_image(
            gather_batch, None, 0, 1, 0.0, draw_gt=draw_gt
        )
        gather_text = f"{origin_name} @ Shot{int(row['ShotId'])}, Rec{int(row['ReceiverId'])}, Id{gather_id}"
        gather_image = draw_utils.draw_big_text_at_top_right(
            gather_image,
            text=gather_text,
            font_scale=2.5,
            font_thickness=4,
            font_color=(12, 242, 12) if gather_is_valid else (12, 12, 224),
        )
        print(
            f"{curr_df_row_idx:06d}/{len(gather_df):06d}:  {gather_text}, Valid={gather_is_valid}"
        )

        original_image_height, original_image_width, _ = gather_image.shape

        image_height = height_magnification * int(
            image_width / original_image_width * original_image_height
        )
        dsize = (image_width, image_height)
        gather_image = cv.resize(
            gather_image, dsize=dsize, fx=2, fy=2, interpolation=cv.INTER_NEAREST
        )
        cv.imshow("test", gather_image)
        key = cv.waitKeyEx(0)
        if (
            key == ord(" ") or key == 32
        ):  # 'spacebar' means toggle groundtruth display on/off
            draw_gt = not draw_gt
        elif key == ord("q") or key == 27:  # 'q' or 'escape' means quit (and save)
            break
        elif (
            key == ord("y") or key == 13 or key == 65477
        ):  # 'y' or 'enter' means toggle valid/invalid
            if gather_is_valid:
                bad_gather_map[origin_name][gather_id] = dict(
                    ShotId=int(row["ShotId"]), ReceiverId=int(row["ReceiverId"])
                )
            else:
                if gather_id in bad_gather_map[origin_name]:
                    del bad_gather_map[origin_name][gather_id]
            gather_is_valid = not gather_is_valid
            curr_df_row_idx = min(curr_df_row_idx + 1, len(gather_df) - 1)
            draw_gt = True
        elif (
            key == ord("a") or key == 65361
        ):  # 'a' or 'left arrow' (with Qt window) goes back one
            curr_df_row_idx = max(curr_df_row_idx - 1, 0)
            draw_gt = True
        elif (
            key == ord("d") or key == 65363
        ):  # 'd' or 'right arrow' (with Qt window) goes forward one
            curr_df_row_idx = min(curr_df_row_idx + 1, len(gather_df) - 1)
            draw_gt = True
        if (
            curr_steps > 100
        ):  # auto-save the current list every 100 steps, no matter what
            print("Saving all flagged gathers as a precaution...")
            with open(output_yaml_path, "w") as fd:
                yaml.dump(bad_gather_map, fd)
            curr_steps = 0

    print("Saving all flagged gathers...")
    with open(output_yaml_path, "w") as fd:
        yaml.dump(bad_gather_map, fd)
    print("All done.")
