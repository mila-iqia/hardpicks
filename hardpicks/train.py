"""This module contains training-related functionalities for PyTorch Lightning.

This file is not meant to be executed directly; it should instead be imported in other modules.
See `main.py` in the same folder for an executable script that can be used to train models.
"""

import glob
import logging
import os
import typing

import pytorch_lightning
import pytorch_lightning.callbacks
import pytorch_lightning.loggers
import torch
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler

from hardpicks.metrics.early_stopping_utils import get_early_stopping_metric_mode
from hardpicks.utils.hp_utils import check_and_log_hp
from hardpicks.utils.prediction_utils import get_data_dump_path

logger = logging.getLogger(__name__)


def create_callbacks(
    experiment_dir: typing.AnyStr,
    run_name: typing.AnyStr,
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    use_progress_bar: bool,
):
    """Creates and returns the list of checkpoint-related callbacks for pt-lightning."""
    callback_dictionary = dict()
    check_and_log_hp(["early_stop_metric", "patience"], hyper_params)
    if hyper_params["early_stop_metric"] not in ["None", "none", "", None]:
        early_stop_metric = hyper_params["early_stop_metric"]
        if "early_stop_metric_mode" in hyper_params:
            early_stop_metric_mode = hyper_params["early_stop_metric_mode"]
        else:
            early_stop_metric_mode = get_early_stopping_metric_mode(early_stop_metric)
        callback_dictionary['early_stopping'] =\
            pytorch_lightning.callbacks.EarlyStopping(
            monitor=early_stop_metric,
            patience=hyper_params["patience"],
            verbose=use_progress_bar,
            mode=early_stop_metric_mode,
        )
        callback_dictionary['best_checkpoint'] = \
            pytorch_lightning.callbacks.ModelCheckpoint(
            dirpath=experiment_dir,
            filename=f"{run_name}." + "best-{epoch:03d}-{step:06d}",
            monitor=early_stop_metric,
            verbose=use_progress_bar,
            mode=early_stop_metric_mode,
        )

    callback_dictionary['last_checkpoint'] = \
        pytorch_lightning.callbacks.ModelCheckpoint(
        dirpath=experiment_dir,
        filename=f"{run_name}." + "last-{epoch:03d}-{step:06d}",
        verbose=use_progress_bar,
    )
    # Commenting out the rich progress bar in order to default back to the
    # internal TQDM progress bar. Rich does not show progress when output
    # is redirected to log, which makes it hard to gauge job progress.
    # callback_dictionary['progress_bar'] = CustomRichProgressBar()

    return callback_dictionary


def train(
    model: pytorch_lightning.LightningModule,
    datamodule: pytorch_lightning.LightningDataModule,
    experiment_dir: typing.AnyStr,
    run_name: typing.AnyStr,
    use_progress_bar: bool,
    mlf_logger: pytorch_lightning.loggers.MLFlowLogger,
    tbx_logger: pytorch_lightning.loggers.TensorBoardLogger,
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    use_gpu_if_available: bool = True,
) -> float:
    """Training loop implementation; creates callbacks and offloads everything to pt-lightning."""
    check_and_log_hp([
        "trainer_checkpoint",
        "precision",
        "log_every_n_steps",
        "num_sanity_val_steps",
        "set_benchmark",
        "set_deterministic",
    ], hyper_params)
    assert ("max_steps" in hyper_params) != ("max_epochs" in hyper_params), \
        "the hyperparameter config should provide either 'max_steps' or 'max_epochs' (and not both!)"
    max_length_key = "max_steps" if "max_steps" in hyper_params else "max_epochs"
    check_and_log_hp([max_length_key], hyper_params)

    # Convenience input to only run the predict phase if the number of epochs is larger than some
    # threshold. This is useful when doing ASHA HP tuning: low resource initial trials are mostly
    # throwaway.
    run_predict_phase_max_epoch_threshold = hyper_params.get("run_predict_phase_max_epoch_threshold", None)
    if run_predict_phase_max_epoch_threshold is not None:
        assert "run_predict_phase" not in hyper_params, \
            "Only one of 'run_predict_phase' or 'run_predict_phase_max_epoch_threshold' should be specified."

        assert type(run_predict_phase_max_epoch_threshold) is int, \
            "Parameter 'run_predict_phase_max_epoch_threshold' should be an integer"
        if hyper_params[max_length_key] >= run_predict_phase_max_epoch_threshold:
            run_predict_phase = True
        else:
            run_predict_phase = False
    else:
        run_predict_phase = hyper_params.get("run_predict_phase", True)
        assert type(run_predict_phase) is bool, "Parameter 'run_predict_phase' should be a boolean"

    # Sanity check the predict_phase configuration BEFORE running the job, to avoid a bad surprise at the end
    # of training
    if 'predict_phase' in hyper_params:
        predict_phase_config = hyper_params['predict_phase']
        assert type(predict_phase_config) == dict, "the 'predict_phase' parameter should contain a dictionary"
        keys = predict_phase_config.keys()
        allowed_keys = {'predict_on_train_data', 'predict_on_valid_data', 'predict_on_test_data'}
        assert set(predict_phase_config.keys()).issubset(allowed_keys), \
            f"The only allowed keys for the 'predict_phase' parameter dictionary are {allowed_keys}"

        for key in keys:
            assert type(predict_phase_config[key]) == bool, f"The parameter {key} should be True or False"

    # first, determine which checkpoint to resume from (if any), if we're not going from scratch
    resume_from_checkpoint = None
    trainer_checkpoint = hyper_params["trainer_checkpoint"]
    if not trainer_checkpoint or trainer_checkpoint == "none" or trainer_checkpoint == "None":
        logger.info("will not load any pre-existing checkpoint.")
    elif trainer_checkpoint == "latest":
        last_models = glob.glob(os.path.join(experiment_dir, f"{run_name}.last-*"))
        if len(last_models) > 1:
            # we've got multiple checkpoints, probably a crash-resume-crash session, just pick the latest
            last_models = sorted(last_models)  # these should be tagged by epoch/step, string-sorting is OK
            logger.info(f"found multiple checkpoints, will resume from latest: {last_models[-1]}")
            resume_from_checkpoint = last_models[-1]
        elif len(last_models) == 1:
            logger.info(f"resuming training from: {last_models[0]}")
            resume_from_checkpoint = last_models[0]
        else:
            logger.info("no model found - starting training from scratch")

    # next, create the list of callbacks needed for checkpointing and logging
    callbacks_dictionary = create_callbacks(
        experiment_dir=experiment_dir,
        run_name=run_name,
        hyper_params=hyper_params,
        use_progress_bar=use_progress_bar,
    )

    # finally, forward the last few hyperparameters we need and create the actual trainer object

    profiling_type = hyper_params.get("profiling", None)
    profiling_default_path = os.path.join(experiment_dir, "profiling.out")
    profiling_output_path = hyper_params.get("profiling_output_path", profiling_default_path)

    if profiling_type == 'simple':
        profiler = SimpleProfiler(output_filename=profiling_output_path)
    elif profiling_type == 'advanced':
        profiler = AdvancedProfiler(output_filename=profiling_output_path)
    else:
        profiler = None

    visible_gpu_count = torch.cuda.device_count()
    logger.info(f"visible gpu count: {visible_gpu_count}")
    if visible_gpu_count == 0:
        logger.info("no available GPU, will run on CPU!")
        use_gpu_if_available = False
    if "use_gpu_if_available" in hyper_params and not hyper_params["use_gpu_if_available"]:
        use_gpu_if_available = False

    trainer = pytorch_lightning.Trainer(
        logger=[mlf_logger, tbx_logger],
        callbacks=list(callbacks_dictionary.values()),
        resume_from_checkpoint=resume_from_checkpoint,
        accelerator="gpu" if use_gpu_if_available else "cpu",
        gpus=visible_gpu_count if use_gpu_if_available else 0,
        auto_select_gpus=True if use_gpu_if_available else False,
        precision=hyper_params["precision"] if use_gpu_if_available else 32,
        log_every_n_steps=hyper_params["log_every_n_steps"],
        num_sanity_val_steps=hyper_params["num_sanity_val_steps"],
        benchmark=hyper_params["set_benchmark"],
        deterministic=hyper_params["set_deterministic"],
        profiler=profiler,
        **{max_length_key: hyper_params[max_length_key]},
    )

    logger.info("Starting data module setup...")
    datamodule.setup()  # this might take a while with big datasets & exhaustive parsers
    logger.info("Data setup complete!")

    # Make it possible to explicitly specify to skip the training phase
    run_train_phase = hyper_params.get("run_train_phase", True)

    if run_train_phase and datamodule.train_dataloader() is not None:
        # if we have a valid train data loader, it probably means we need to train...
        # (otherwise, this branch will be skipped, and we'll go directly to predictions)
        trainer.fit(model, datamodule=datamodule)
    else:
        # Let's still make sure that the trainer has access to the datamodule internally;
        # this information might be necessary to do a setup set at prediction time.
        trainer.datamodule = datamodule

    if 'best_checkpoint' in callbacks_dictionary:
        checkpoint_callback = callbacks_dictionary['best_checkpoint']
    else:
        checkpoint_callback = callbacks_dictionary['last_checkpoint']

    best_model_path = checkpoint_callback.best_model_path

    if best_model_path:  # if we actually got a 'best' checkpoint, reload it before predictions
        prediction_model = model.__class__.load_from_checkpoint(best_model_path)
        setattr(prediction_model, "_mlf_logger", model._mlf_logger)
        setattr(prediction_model, "_tbx_logger", model._tbx_logger)
    else:
        logger.warning(
            "could not locate an existing 'best' checkpoint post-training; "
            "will do predictions/final testing with the latest state of the trained model directly"
        )
        prediction_model = model

    # Run test phase, if specified.
    run_test_phase = hyper_params.get("run_test_phase", False)
    if run_test_phase and datamodule.test_dataloader() is not None:
        _ = trainer.test(model=prediction_model, dataloaders=datamodule.test_dataloader())

    if run_predict_phase:
        predict_phase_config = hyper_params.get("predict_phase", None)

        if predict_phase_config is None:
            # Assume we'll predict on all datasets
            loader_tuples = [
                ("train", datamodule.train_dataloader()),
                ("valid", datamodule.val_dataloader()),
                ("test", datamodule.test_dataloader()),
            ]
        else:
            loader_tuples = []
            if predict_phase_config.get('predict_on_train_data', False):
                loader_tuples.append(("train", datamodule.train_dataloader()))
            if predict_phase_config.get('predict_on_valid_data', False):
                loader_tuples.append(("valid", datamodule.val_dataloader()))
            if predict_phase_config.get('predict_on_test_data', False):
                loader_tuples.append(("test", datamodule.test_dataloader()))

        logger.info("#########         Entering Predict Phase         #################")
        # time to "predict" on all the data to log the final results/metrics now; we'll set the
        # appropriate secret attribute on the model to make this happen (see the `predict`-related
        # functions in models/base.py for more information)

        for dataloader_type, dataloader in loader_tuples:
            if dataloader is not None:
                logger.info(f"      Predict Phase  : Starting PREDICT for {dataloader_type}")

                # the data dump will be done next to the model checkpoint (in the experiment directory);
                # MLFlow will log it as an artifact to a better location after the fact, and sigopt
                # will be provided the final metrics at this stage as well (if enabled)
                output_data_dump_path = \
                    get_data_dump_path(best_model_path, dataloader_type, experiment_dir)
                assert hasattr(prediction_model, "predict_eval_output_path"), "missing secret attrib?"
                prediction_model.predict_eval_output_path = output_data_dump_path
                # note: we don't actually need the predictions here (just focus on eval results)
                _ = trainer.predict(model=prediction_model, dataloaders=dataloader)
                logger.info(f"      Predict Phase  : Completed PREDICT for {dataloader_type}")

        logger.info("#########         Exiting Predict Phase         #################")

    # check for a end-of-training hook that pytorch-lightning does NOT call automatically (it's our own!)
    if hasattr(model, "on_experiment_end") and callable(model.on_experiment_end):
        # this might do some final cleaning up or report final test results elsewhere...
        model.on_experiment_end()

    # finally, we need to push the early stopping criterion (score) back to the caller for orion
    early_stopping = callbacks_dictionary.get("early_stopping", None)
    if early_stopping is not None:
        return float(early_stopping.best_score.cpu().numpy())
    return 1e10  # if we don't have any such criterion... well, we have to cheese it somehow
