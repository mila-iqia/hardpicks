#!/usr/bin/env python
"""Entrypoint for training deep neural networks using PyTorch-Lightning using config files."""

import argparse
import glob
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
import typing

import deepdiff
import mlflow
import orion
import orion.client
import pytorch_lightning
import pytorch_lightning.loggers
import pytorch_lightning.utilities
import yaml
from mlflow.entities import RunStatus

import torch

from hardpicks import PATH_TO_GPU_MONITORING_SCRIPT
from hardpicks.data.data_loader import create_data_module
from hardpicks.metrics.early_stopping_utils import (
    get_name_and_sign_of_orion_optimization_objective,
)
from hardpicks.train import train
from hardpicks.utils.hp_utils import (
    check_and_log_hp,
    check_config_contains_sigopt_tokens,
    get_run_name,
    load_mlflow_run_info,
    write_mlflow_run_info,
    replace_placeholders,
    log_run_failure,
)
from hardpicks.models.model_loader import load_model
from hardpicks.utils.file_utils import rsync_folder
from hardpicks.utils.logging_utils import (
    LoggerWriter,
    log_exp_details,
)
from hardpicks.utils.multiproc_utils import set_multiprocessing_start_method
from hardpicks.utils.reproducibility_utils import set_seed

logger = logging.getLogger(__name__)


def setup_arg_parser(
    with_output_args: bool = True,  # use for other read-only scripts w/ similar args
) -> argparse.ArgumentParser:
    """Prepares and returns the argument parser for the CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Config file to parse (in yaml format).", required=True,
    )
    parser.add_argument(
        "--data", help="Path to the data expected by the data module.", required=True,
    )
    parser.add_argument(
        "--tmp-folder",
        help="Path to a temporary folder that will be as working directory to minimize filesystem I/O "
        "delays. The entirety of the `data` folder will first be copied here, then all experiment "
        "logs and artifacts will be created here, and finally the content of this folder will be "
        "returned to the real `output` directory at the end of the session. Note that using this "
        "temporary folder with the default mlflow path will throw an error, as it cannot be moved.",
    )
    parser.add_argument(
        "--disable-progressbar",
        action="store_true",
        help="Used to disable the display of the progress bar while iterating through batches. Useful "
        "to minimize the size of log files and to reduce useless outputs in long experiments.",
    )
    parser.add_argument(
        "--use-gpu",
        help="Toggles whether to automatically use gpu(s) if available; PyTorch-Lightning will"
        " automatically figure out which devices to use.",
        default=True,
    )
    parser.add_argument(
        "--disable-logger-writer",
        action="store_true",
        help="Boolean argument to set whether the output will be captured by the LoggerWriter "
        "class and redirected to a log. This is a debugging option and should not be used"
        "in production.",
        default=False,
    )
    if with_output_args:
        parser.add_argument(
            "--output",
            help="Path to the folder where experiment outputs should be saved. A directory for this "
                 "particular experiment will be created there using the ORION-provided experiment name "
                 "or the user-provided one found in the hyper parameter configuration file. Also, by "
                 "default, the tensorboard and mlflow output folders will automatically be created there.",
            required=True,
        )
        parser.add_argument(
            "--erase-output-dir-if-exists",
            action="store_true",
            help="Toggle whether to erase the experiment output directory if it already exists at launch."
                 "Useful for debugging configs while repeatedly launching from-scratch experiments.",
        )
        parser.add_argument(
            "--tensorboard-output",
            help="Path to the folder where tensorboard experiment log directories should be saved. By "
                 "default, if no path is provided, a `tensorboard` folder will be created in the output "
                 "directory.",
        )
        parser.add_argument(
            "--mlflow-output",
            help="Path to the folder where mlflow experiment log directories should be saved. By "
                 "default, if no path is provided, a `mlflow` folder will be created in the output "
                 "directory.",
        )

    return parser


def prepare_experiment(
    config_file_path: typing.AnyStr,
    output_dir_path: typing.AnyStr,
    erase_output_dir_if_exists: bool = False,
    read_only: bool = False,
):
    """Prepares the experiment by parsing/validating meta parameters and preparing folders."""
    # first, load the hyperparameter configuration file and validate the stuff we need right now
    with open(config_file_path, "r") as stream:
        hyper_params = yaml.load(stream, Loader=yaml.FullLoader)

    # next, determine which experiment name we'll truly be using and create its corresponding dir
    exp_name = None
    if orion.client.cli.IS_ORION_ON:
        assert not read_only, "should not be using ORION if in read-only (predict) mode!"
        exp_name = os.getenv("ORION_EXPERIMENT_NAME", default=None)
    if exp_name is None:
        assert (
            "exp_name" in hyper_params and hyper_params["exp_name"]
        ), "missing experiment name (`exp_name`) in hyper parameter configuration file"
        exp_name = hyper_params["exp_name"]
    experiment_dir = os.path.join(output_dir_path, exp_name)
    if erase_output_dir_if_exists and os.path.isdir(experiment_dir):
        assert not read_only, "should not have output dir erase flag on in read_only mode!"
        shutil.rmtree(experiment_dir)
    if read_only:
        assert os.path.isdir(experiment_dir), \
            f"in read-only/predict mode, experiment dir should already exist at: {experiment_dir}"
        assert not check_config_contains_sigopt_tokens(hyper_params=hyper_params), \
            "in read-only/predict mode, configs should not contain any sigopt search token!"
    else:
        os.makedirs(experiment_dir, exist_ok=True)
    run_name = get_run_name(hyper_params)
    hyper_params = replace_placeholders(hyper_params)

    # keep an intact copy of the original configuration file or verify the previous backup
    config_file_backup_path = os.path.join(experiment_dir, f"{run_name}.config.yaml")
    backup_config_exists = os.path.exists(config_file_backup_path)
    if backup_config_exists:
        with open(config_file_backup_path, "r") as stream:
            hyper_params_backup = yaml.load(stream, Loader=yaml.FullLoader)
        hyper_params_backup = replace_placeholders(hyper_params_backup)
        # better crash the experiment here, otherwise it's mlflow that will crash it later
        assert not deepdiff.DeepDiff(hyper_params_backup, hyper_params), (
            f"INCOMPATIBLE BACKUP CONFIG ALREADY PRESENT IN EXPERIMENT DIRECTORY!\n"
            f"\t...conflicting config file path: {config_file_backup_path}\n"
            "\t...you must update your experiment/run name, or erase the previous training result!"
        )
    else:
        shutil.copyfile(config_file_path, config_file_backup_path)

    # next, make sure RNG-related stuff is set if needed
    if hyper_params["seed"] is not None:
        set_seed(
            seed=hyper_params["seed"],
            set_deterministic=hyper_params["set_deterministic"],
            set_benchmark=hyper_params["set_benchmark"],
        )
    set_multiprocessing_start_method(hyper_params)

    return run_name, exp_name, experiment_dir, hyper_params, config_file_backup_path


def prepare_loggers(
    tensorboard_save_dir: typing.AnyStr,
    experiment_name: typing.AnyStr,
    experiment_dir: typing.AnyStr,
    run_name: typing.AnyStr,
):
    """Instantiates and returns the mlflow/tensorboard experiment loggers to use."""
    # first, prepare the mlflow logger with the proper tracking uri (from current run)
    mlf_logger = pytorch_lightning.loggers.MLFlowLogger(
        experiment_name=experiment_name, tracking_uri=mlflow.get_tracking_uri(),
    )
    mlf_logger._run_id = mlflow.active_run().info.run_id

    # next, prepare the tensorboard logger with a new local directory (if needed!)
    os.makedirs(tensorboard_save_dir, exist_ok=True)
    tbx_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=tensorboard_save_dir,
        name=experiment_name + "/" + run_name,
        default_hp_metric=False,
    )

    # to log all prints to a text file inside the output directory:
    console_log_file = os.path.join(experiment_dir, "console.log")
    handler = logging.handlers.WatchedFileHandler(console_log_file)
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    return mlf_logger, tbx_logger


def main(args: typing.Optional[typing.Any] = None):
    """Main entry point of the program.

    Note:
        To get information on the expected arguments for this entrypoint, execute the script
        directly using python as such:
            python <PATH_TO_THIS_REPO>/hardpicks/main.py --help
    """
    parser = setup_arg_parser()
    args = parser.parse_args(args)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Mechanism to turn this off while debugging; it
    #  prevents the interactive ipython shell from getting control.
    if not args.disable_logger_writer:
        sys.stdout = LoggerWriter(logger.info)
        sys.stderr = LoggerWriter(logger.warning)

    os.makedirs(args.output, exist_ok=True)
    root_data_dir = args.data

    # if a temp folder is provided, we'll also put all the output there until the end of the session
    if args.tmp_folder is not None:
        assert (
            args.mlflow_output is not None
        ), "cannot leave mlflow output path as default w/ temp dir"
        # we'll give our temp folder a distinctive subdirectory to avoid conflicts w/ other projects
        tmp_folder = os.path.join(args.tmp_folder, "hardpicks")
        os.makedirs(tmp_folder, exist_ok=True)
        # first, copy the data from its origin to the temp directory (this will help on the cluster!)
        rsync_folder(args.data, tmp_folder)
        local_data_dir = os.path.join(
            tmp_folder, os.path.basename(os.path.normpath(root_data_dir))
        )
        # next, prepare the folder where we'll actually be saving our experiment results
        output_dir = os.path.join(tmp_folder, "output")
        os.makedirs(output_dir, exist_ok=True)
    else:
        local_data_dir = root_data_dir
        output_dir = args.output

    # set up the output paths for the loggers if we have not specified them
    if args.mlflow_output is None:
        args.mlflow_output = os.path.join(output_dir, "mlruns")
    if args.tensorboard_output is None:
        args.tensorboard_output = os.path.join(output_dir, "tensorboard")

    # prepare the experiment directory structure and load the hyperparameters (without checks yet!)
    (
        run_name,
        exp_name,
        experiment_dir,
        hyper_params,
        config_file_backup_path,
    ) = prepare_experiment(
        config_file_path=args.config,
        output_dir_path=output_dir,
        erase_output_dir_if_exists=args.erase_output_dir_if_exists,
    )

    # add tmp_folder to hyper_params
    hyper_params["tmp_folder"] = args.tmp_folder

    logger.info(
        f"Experiment '{exp_name}' will output to path: {os.path.abspath(experiment_dir)}"
    )
    pre_existing_checkpoint_paths = glob.glob(os.path.join(experiment_dir, "*.ckpt"))

    # if the experiment/run already existed, make sure we load the id before anything else
    mlflow.set_tracking_uri(os.path.abspath(args.mlflow_output))
    mlflow.set_experiment(exp_name)
    # DISABLE the autolog, or else it interferes with our own logging
    mlflow.pytorch.autolog(log_models=False, disable=True)
    mlflow_run_id = None
    mlflow_run_info_path = os.path.join(experiment_dir, f"{run_name}.mlflow.yaml")
    if os.path.exists(mlflow_run_info_path):
        mlflow_run_info = load_mlflow_run_info(mlflow_run_info_path)
        assert mlflow_run_info["experiment_name"] == exp_name
        mlflow_run_id = mlflow_run_info["run_id"]

    # Start monitoring GPU if so requested.
    gpu_monitoring_process = None
    if torch.cuda.device_count() > 0 and hyper_params.get("track_gpu_stats", False):
        output_gpu_monitoring_file_path = os.path.join(
            experiment_dir, "gpu_monitoring.csv"
        )
        env = os.environ.copy()
        # Here we launch an independent process to get a more accurate (ie, asynchronous, time periodic) reading
        # of GPU usage. The GPU monitoring callback in pytorch lightning is event-based, and so only measures in
        # a biased way.
        # It takes a bit of time for the logging to begin and for the gpu_monitoring.csv file to appear. The
        # file might not b generated during very short runs.
        logger.info(f"Logging GPU stats to file : {output_gpu_monitoring_file_path}")
        gpu_monitoring_process = subprocess.Popen(
            [
                "python",
                PATH_TO_GPU_MONITORING_SCRIPT,
                "--output_csv_file_path",
                output_gpu_monitoring_file_path,
            ],
            env=env,
        )

    # time to actually set up the experiment/run now
    with mlflow.start_run(run_id=mlflow_run_id, run_name=run_name):
        write_mlflow_run_info(mlflow_run_info_path)
        mlflow.log_artifact(config_file_backup_path, "configs")

        # prepare the loggers & initialize the mlflow run
        mlf_logger, tbx_logger = prepare_loggers(
            tensorboard_save_dir=args.tensorboard_output,
            experiment_name=exp_name,
            experiment_dir=experiment_dir,
            run_name=run_name,
        )

        # we can now check hps that couldn't be checked earlier since the calls log them to mlflow
        log_exp_details(os.path.realpath(__file__), root_data_dir)
        check_and_log_hp(
            ["exp_name", "seed", "set_deterministic", "set_benchmark"], hyper_params
        )

        # it's finally time to create the data module & the lightning module
        data_module = create_data_module(local_data_dir, hyper_params)
        model = load_model(hyper_params)
        model_summary = pytorch_lightning.utilities.model_summary.summarize(model, max_depth=4)
        logger.info(f"Model summary:\n{model_summary}")

        # give a direct backdoor access to tbx/mlf loggers for prototyping
        setattr(model, "_tbx_logger", tbx_logger)
        setattr(model, "_mlf_logger", mlf_logger)

        # send all these lovely things to the yard to play
        orion_bad_trial_objective = 1e10
        cuda_out_of_memory = False
        try:
            best_dev_metric = train(
                model=model,
                datamodule=data_module,
                experiment_dir=experiment_dir,
                run_name=run_name,
                use_progress_bar=not args.disable_progressbar,
                mlf_logger=mlf_logger,
                tbx_logger=tbx_logger,
                hyper_params=hyper_params,
                use_gpu_if_available=args.use_gpu,
            )
        except Exception as err:
            log_run_failure(err)
            if (
                isinstance(err, RuntimeError)
                    and orion.client.cli.IS_ORION_ON
                    and "CUDA out of memory" in str(err)
            ):
                logger.error("CUDA out of memory: Will report a bad score to Orion to avoid this config")
                best_dev_metric = -orion_bad_trial_objective
                cuda_out_of_memory = True
            else:
                raise err

        # we'll just log any new checkpoints created in this run as artifacts
        final_checkpoint_paths = glob.glob(os.path.join(experiment_dir, "*.ckpt"))
        for ckpt_path in final_checkpoint_paths:
            if ckpt_path not in pre_existing_checkpoint_paths:
                mlflow.log_artifact(os.path.abspath(ckpt_path), "checkpoints")

        if cuda_out_of_memory:
            # Explicitly tell mlflow we crashed to avoid confusing reporting in the mlflow ui.
            mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

        # don't forget to tell orion how good we just did!
        if hyper_params["early_stop_metric"] not in ["None", "none", "", None]:
            (
                optimization_objective_name,
                optimization_sign,
            ) = get_name_and_sign_of_orion_optimization_objective(
                early_stopping_metric=hyper_params["early_stop_metric"],
                mode=hyper_params.get("early_stop_metric_mode", None),
            )
        else:
            optimization_objective_name = "unknown"
            optimization_sign = 1

        if orion.client.cli.IS_ORION_ON:
            logger.info(" Reporting Results to ORION...")
            orion.client.report_results(
                [
                    dict(
                        name=optimization_objective_name,
                        type="objective",
                        # note the minus - cause orion is always trying to minimize (cit. from the guide)
                        value=optimization_sign * float(best_dev_metric),
                    ),
                ]
            )
            logger.info(" Done reporting Results to ORION.")

    if args.tmp_folder is not None:
        # note: if the destination experiment folder already exists, it will be overwritten...
        rsync_folder(output_dir + os.path.sep, args.output)

    if gpu_monitoring_process and gpu_monitoring_process.poll() is not None:
        gpu_monitoring_process.terminate()


if __name__ == "__main__":
    main()
