#!/usr/bin/env python
"""Utility/debug script used to test different data loader parameters and check its performance."""

import argparse
import logging.handlers
import pathlib
import tempfile
import time
import typing

import cv2 as cv
import mock
import torch.cuda

import hardpicks.data.data_loader as main_loader
import hardpicks.main as main
import hardpicks.metrics.eval_loader as eval_loader
import hardpicks.models.fbp.utils as fbp_model_utils

logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    """Prepares and returns the argument parser for the CLI entrypoint."""
    # note: the basic args below are the same used in the real 'main' CLI module
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file to parse (in yaml format).", required=True)
    parser.add_argument("--data", help="Path to the data expected by the data module.", required=True)
    parser.add_argument("--display", help="Display input/target data", action="store_true", default=False)
    parser.add_argument("--run-model", help="Run batches thru model", action="store_true", default=False)
    parser.add_argument("--run-metrics", help="Run preds thru metrics", action="store_true", default=False)
    parser.add_argument("--max-iters", help="Max data loader iteration count", default=100)
    parser.add_argument("--orig-output", help="Original output dir for pre-existing experiments", default=None)
    return parser


def _get_checkpoint_if_one_exists(experiment_dir, run_name) -> typing.Optional[typing.AnyStr]:
    """Returns a checkpoint for the specified experiment/run name combination, if possible."""
    best_model_ckpts = sorted(list(pathlib.Path(experiment_dir).glob(f"{run_name}.best-*.ckpt")))
    if len(best_model_ckpts):
        return str(best_model_ckpts[-1])
    last_model_ckpts = sorted(list(pathlib.Path(experiment_dir).glob(f"{run_name}.last-*.ckpt")))
    if len(last_model_ckpts):
        return str(last_model_ckpts[-1])
    return None


def _upload_all_tensors(batch, device):
    """Uploads all tensors in the loaded batch to the specified device."""
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)


def _do_one_iteration(batch, batch_idx, model, optimizer, evaluator):
    """Runs a forward+backward+optim-step pass (i.e. one full iteration)."""
    _upload_all_tensors(batch, model.device)
    should_optimize_here = not hasattr(model, "automatic_optimization") or model.automatic_optimization
    if should_optimize_here:
        optimizer.zero_grad()
    preds, loss, _ = model._generic_step(batch, batch_idx, evaluator)
    if should_optimize_here:
        loss.backward()
        optimizer.step()
    return preds


def _display_fbp_batch(batch, preds, hyper_params):
    """Renders and display the images inside a first-break-picking data batch."""
    for gather_idx in range(batch["batch_size"]):
        gather_image = fbp_model_utils.generate_pred_image(
            batch=batch,
            raw_preds=preds,
            batch_gather_idx=gather_idx,
            segm_class_count=hyper_params["segm_class_count"],
            segm_first_break_prob_threshold=hyper_params["segm_first_break_prob_threshold"],
            draw_prior=True,
        )
        gather_image = cv.resize(
            gather_image,
            dsize=(-1, -1),
            fx=2,
            fy=2,
            interpolation=cv.INTER_NEAREST,
        )
        cv.imshow("test", gather_image)
        print(f"{batch['origin'][gather_idx]} @ {batch['gather_id'][gather_idx]}")
        cv.waitKey(0)


def loader_profiler(args: typing.Optional[typing.Any] = None):
    """Main entry point of the profiling program."""
    parser = setup_arg_parser()
    args = parser.parse_args(args)
    local_data_dir = args.data

    with tempfile.TemporaryDirectory() as tmpdir:
        run_name, exp_name, experiment_dir, hyper_params, config_file_backup_path = \
            main.prepare_experiment(
                config_file_path=args.config,
                output_dir_path=tmpdir,
            )

        hyper_params["num_workers"] = 0  # keeps iterations more smooth (albeit slower overall)

        data_module = main_loader.create_data_module(local_data_dir, hyper_params)
        data_module.prepare_data()
        data_module.setup()
        # we'll only run the training data pipeline here
        data_loader = data_module.train_dataloader()
        assert data_loader is not None
        assert len(data_loader) > 0
        init_batch = next(iter(data_loader))  # to properly initialize the model on 1st iteration

        model, evaluator = None, mock.Mock()
        if args.run_model:
            if hyper_params.get("model_checkpoint", None) is None and args.orig_output is not None:
                orig_experiment_dir = pathlib.Path(args.orig_output) / exp_name
                hyper_params["model_checkpoint"] = _get_checkpoint_if_one_exists(
                    experiment_dir=orig_experiment_dir,
                    run_name=run_name,
                )
            from hardpicks.models.model_loader import load_model
            model = load_model(hyper_params)
            model.datamodule = data_module  # for internal usage, if necessary
            model = model.train()
            if torch.cuda.is_available():
                model = model.cuda()
            optimizer = model.configure_optimizers()
            if args.run_metrics:
                evaluator = eval_loader.get_evaluator(hyper_params)
            # initialize internal stuff here to make sure 1st iteration timing will be significant
            _do_one_iteration(init_batch, -1, model, optimizer, evaluator)

        iters, max_iters = 0, args.max_iters
        init_time = time.time()
        prev_iter_end_time = init_time
        # for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, total=len(data_loader))):
        for batch_idx, batch in enumerate(data_loader):
            print(f"running batch {batch_idx}/{len(data_loader)}")
            preds = None
            if args.run_model:
                pre_model_iter_time = time.time()
                preds = _do_one_iteration(batch, batch_idx, model, optimizer, evaluator)
                model_iter_time = time.time() - pre_model_iter_time
                loop_iter_time = time.time() - prev_iter_end_time
                model_iter_ratio = model_iter_time / loop_iter_time
                print(f"\tmodel iter = {model_iter_time:.3f} sec  ({model_iter_ratio:.1%} of loop iter)")
                prev_iter_end_time = time.time()
            if args.display:
                if hyper_params["module_type"] == "FBPDataModule":
                    _display_fbp_batch(batch, preds, hyper_params)
                else:
                    raise AssertionError("unrecognitzed data module type")
            iters += 1
            if max_iters is not None and iters > max_iters:
                break
        tot_time = time.time() - init_time
        print(f"all done in {tot_time} seconds  ({iters / tot_time} it/sec)")


if __name__ == "__main__":
    loader_profiler()
