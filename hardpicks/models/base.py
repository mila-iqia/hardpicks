import abc
import logging
import mock
import os
import typing

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.utilities.types as pl_types
import torch.optim

import torch
import torch.utils.data
import torch.nn.functional

import hardpicks.metrics.base as eval_base
import hardpicks.metrics.eval_loader as eval_loader
import hardpicks.models.optimizers as optimizers
import hardpicks.models.schedulers as schedulers
import hardpicks.utils.hp_utils as hp_utils
import hardpicks.utils.prediction_utils as pred_utils

sigopt = mock.Mock()  # Let's remove sigopt for now.

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """Base PyTorch-Lightning model interface."""

    def __init__(
        self,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any]
    ):
        """Validates+logs model hyperparameters and sets up the metrics."""
        super().__init__()
        hp_utils.check_and_log_hp([
            "update_scheduler_at_epochs",  # defines whether scheduler updates at epochs or steps
            "optimizer_type",
            "scheduler_type",
            "max_epochs",  # might be needed by the scheduler!
        ], hyper_params)
        self.save_hyperparameters(hyper_params)  # they will become available via model.hparams
        # plightning overlaps the 'epochs' of the different loops sometimes, so we need unique objs...
        self.train_evaluator, self.valid_evaluator, self.test_evaluator, self.pred_evaluator = \
            [eval_loader.get_evaluator(hyper_params) for _ in range(4)]
        use_full_metrics_during_training = hyper_params.get("use_full_metrics_during_training", True)
        if not use_full_metrics_during_training:
            # the evaluator might be 'too heavy' to run on each training minibatch; this disables it
            self.train_evaluator = eval_base.NoneEvaluator(hyper_params=None)
        self.update_scheduler_at_epochs = hyper_params["update_scheduler_at_epochs"]
        self.scheduler = None  # we'll manage the scheduler manually through this attribute...
        self.predict_eval_output_path = None  # will be set from outside if we want to dump results
        self.images_to_display = hyper_params.get("images_to_display", 0)
        self.data_ids_to_render_and_log = {}  # updated at 1st epoch start w/ random ids

    def _create_scheduler(self, optimizer):
        """Create scheduler.

        In an ideal world, scheduler creation should happen in the function "configure_optimizers".
        Unfortunately, advanced schedulers need to know how many steps will be taken per epochs;
        this information requires knowing about the train_dataloader, which is not always available
        when "configure_optimizers" is called, in particular when using trainer.fit(model, dataloader=...).

        There seems to be a github issue about this:
        https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-757863689

        and potentially a PR workaround:
        https://github.com/PyTorchLightning/pytorch-lightning/pull/11599

        These are not in the current stable release, so we'll create our own workaround.
        # TODO: review this when the PR above is merged in the stable release of PL.

        This method is meant to be called in "on_training_start", where dataloaders should be available.
        """
        assert hasattr(self, 'trainer') and hasattr(self.trainer, 'num_training_batches'), \
            "The property 'num_training_batches' is not available when '_create_scheduler' is called. Review code!"

        step_count_per_epoch = self.trainer.num_training_batches
        assert type(step_count_per_epoch) == int and step_count_per_epoch > 0, \
            f"num_training_batches is {step_count_per_epoch}, which is not appropriate."

        scheduler = schedulers.get_scheduler(
            scheduler_type=self.hparams["scheduler_type"],
            scheduler_params=self.hparams.get("scheduler_params", {}),
            optimizer=optimizer,
            max_epochs=self.hparams["max_epochs"],
            step_count_per_epoch=step_count_per_epoch,
        )
        if hasattr(scheduler, "must_step_per_iter") and scheduler.must_step_per_iter:
            assert not self.update_scheduler_at_epochs, "invalid scheduler/update freq combo"
        return scheduler

    def configure_optimizers(self):
        """Returns the optimizer(s) to use during training."""
        return optimizers.get_optimizer(
            optimizer_type=self.hparams["optimizer_type"],
            optimizer_params=self.hparams.get("optimizer_params", {}),
            model_params=self.parameters(),
        )

    def is_initialized(self) -> bool:
        """Returns whether the model is ready-to-be-used or not.

        For PyTorch-Lightning-based models, they are essentially always ready, but in some special
        cases, we might want to see some data first. It is up to the derived classes to implement this
        if needed.
        """
        return True

    @abc.abstractmethod
    def _generic_step(
        self,
        batch: typing.Any,
        batch_idx: int,
        evaluator: eval_base.EvaluatorBase,
    ) -> typing.Tuple[typing.Any, torch.Tensor, typing.Dict[typing.AnyStr, float]]:
        """Runs the prediction + evaluation step for training/validation/testing."""
        # note: this function should return the model predictions, the loss, and the metrics
        raise NotImplementedError

    @staticmethod
    def _get_batch_size_from_data_loader(data_loader):
        """Returns the batch size that will be (usually) used by a given data loader."""
        if hasattr(data_loader, 'batch_sampler'):
            return data_loader.batch_sampler.batch_size
        else:
            return data_loader.batch_size

    @staticmethod
    def _get_latest_metric_eval_results(
        prefix: typing.AnyStr,
        losses: typing.Optional[typing.Iterable[torch.Tensor]],
        evaluator: eval_base.EvaluatorBase,
    ) -> typing.Dict[typing.AnyStr, float]:
        """Returns the summarized metric evaluation results contained in the given evaluator."""
        results = {}

        # first, report for each individual category using the proper prefix
        for category in evaluator.get_categories():
            metrics = evaluator.summarize(category_name=category)
            for metric_name, metric_val in metrics.items():
                results[f"{category}/{prefix}/{metric_name}"] = metric_val

        # next, report the overall results without any prefix
        metrics = evaluator.summarize(category_name=None)
        for metric_name, metric_val in metrics.items():
            results[f"{prefix}/{metric_name}"] = metric_val

        # finally, report the average loss for the entire epoch (if not training)
        if prefix != "train" and losses is not None:
            loss_array = [t.item() for t in losses]
            results[f"{prefix}/loss"] = np.mean(loss_array)

        return results

    def _generic_epoch_end(
        self,
        prefix: typing.AnyStr,
        losses: typing.Iterable[torch.Tensor],
        evaluator: eval_base.EvaluatorBase,
    ):
        """Completes the epoch by asking the evaluator to summarize its results to log them."""
        results = self._get_latest_metric_eval_results(
            prefix=prefix,
            losses=losses,
            evaluator=evaluator,
        )
        # Here all the metrics are logged at the end of an epoch.
        # according to
        #   https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-from-a-lightningmodule
        # The signature of the log function is
        #  self.log("my_whatever", whatever, on_step=on_step, on_epoch=on_epoch, ,...)
        # and for on_[train/test/valid]_epoch_end, on_step = False and on_epoch = True.
        # The following logging thus takes care of ALL the epoch level logging for METRICS.
        # Also, the "results" dictionary contains the epoch-averaged loss if we are not in training.
        for metric_name, metric_val in results.items():
            self.log(metric_name, metric_val)
        sigopt.log_checkpoint(results)  # according to the doc, it doesn't need an epoch index either

    def on_train_start(self):
        """This method is called when training starts."""
        super().on_train_start()
        optimizer = self.optimizers()
        if optimizer is None:
            self.scheduler = None
        else:
            # Insure that there is a single optimizer. If not, we are not in Vanilla mode
            # and this method should be overloaded!
            assert isinstance(optimizer, torch.optim.Optimizer), \
                "we currently only support one optimizer for the current impl; override this!"
            self.scheduler = self._create_scheduler(optimizer)

    def on_train_epoch_start(self):
        """Resets the evaluator state before the start of any new training epoch."""
        self.train_evaluator.reset()
        assert self.scheduler is not None, "need to define a scheduler before training!"
        # if we need to pick which data samples to display in advance, do it now
        self._pick_data_ids_to_render_and_log(prefix="train", data_loader=self._get_train_dataloader())

    def on_epoch_start(self):
        """Callback to perform at the start of each epoch."""
        if self.on_gpu:
            torch.cuda.reset_peak_memory_stats(device=self.device)

    def on_epoch_end(self):
        """Callback to perform at the end of each epoch."""
        if self.on_gpu:
            max_mem_mb = torch.cuda.max_memory_allocated(device=self.device) // (1024 * 1024)
            self.log("maximum_cuda_memory_mb", float(max_mem_mb))
            sigopt.log_metric("maximum_cuda_memory_mb", max_mem_mb)
        sigopt.log_metric("completed_epochs", self.current_epoch + 1)

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training data."""
        preds, loss, metrics = self._generic_step(batch, batch_idx, self.train_evaluator)

        # We are logging the training loss at every step: the self.log default values for
        # the 'training_step' call back are on_step = True, on_epoch = False
        self.log("train/loss", loss.item(), prog_bar=True)

        # We are logging learning rate and epoch at every step. These are not "metrics",
        # however, and so there will never be an epoch level logging.
        self.log("train/learning_rate", self.scheduler.get_last_lr()[0])
        self.log("train/epoch", float(self.current_epoch))
        self._render_and_log_data_samples_from_ids(batch, batch_idx, preds, "train")
        if not self.update_scheduler_at_epochs:
            self.scheduler.step()  # this is useful for granular schedulers, e.g. cosine annealer
        # note: returning the predictions in this dict might blow up the memory for long epochs...
        return dict(loss=loss, metrics=metrics)  # loss is required in this dict!

    def training_epoch_end(self, outputs: pl_types.EPOCH_OUTPUT):
        """Completes the epoch by asking the evaluator to summarize its results."""
        losses = [d["loss"] if isinstance(d, dict) else d for d in outputs]
        self._generic_epoch_end(prefix="train", losses=losses, evaluator=self.train_evaluator)
        if self.update_scheduler_at_epochs:
            # following PyTorch 1.1+ design, we step at the end of the training epoch...
            # see https://github.com/pytorch/pytorch/issues/20124 for more info
            self.scheduler.step()

    def on_validation_epoch_start(self):
        """Resets the evaluator state before the start of any new validation epoch."""
        self.valid_evaluator.reset()
        # if we need to pick which data samples to display in advance, do it now
        self._pick_data_ids_to_render_and_log(prefix="valid", data_loader=self._get_val_dataloader())

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation data."""
        preds, loss, metrics = self._generic_step(batch, batch_idx, self.valid_evaluator)
        self._render_and_log_data_samples_from_ids(batch, batch_idx, preds, "valid")
        # note: returning the predictions in this dict might blow up the memory for long epochs...
        return dict(loss=loss, metrics=metrics)

    def validation_epoch_end(self, outputs: pl_types.EPOCH_OUTPUT):
        """Completes the epoch by asking the evaluator to summarize its results."""
        losses = [d["loss"] if isinstance(d, dict) else d for d in outputs]
        self._generic_epoch_end(prefix="valid", losses=losses, evaluator=self.valid_evaluator)

    def on_test_epoch_start(self):
        """Resets the evaluator state before the start of any new testing epoch."""
        self.test_evaluator.reset()
        # if we need to pick which data samples to display in advance, do it now
        self._pick_data_ids_to_render_and_log(prefix="test", data_loader=self._get_test_dataloader())

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing data."""
        preds, loss, metrics = self._generic_step(batch, batch_idx, self.test_evaluator)
        self._render_and_log_data_samples_from_ids(batch, batch_idx, preds, "test")
        # note: returning the predictions in this dict might blow up the memory for long epochs...
        return dict(loss=loss, metrics=metrics)

    def test_epoch_end(self, outputs: pl_types.EPOCH_OUTPUT):
        """Completes the epoch by asking the evaluator to summarize its results."""
        losses = [d["loss"] if isinstance(d, dict) else d for d in outputs]
        self._generic_epoch_end(prefix="test", losses=losses, evaluator=self.test_evaluator)

    def on_predict_epoch_start(self):
        """Resets the evaluator state before the start of a (deploy-time) prediction epoch.

        Note: the model behavior here should be the same as for validation/testing. The difference
        might be the lack of groundtruth for evaluation, but that will be up to the evaluator to
        handle.
        """
        self.pred_evaluator.reset()

    def predict_step(
        self,
        batch: typing.Any,
        batch_idx: int,
        dataloader_idx: typing.Optional[int] = None,
    ) -> typing.Any:
        """Runs a prediction step on new data, returning only the predictions.

        Note: the model behavior here should be the same as for validation/testing. The difference
        might be the lack of groundtruth for evaluation, but that will be up to the model/evaluator
        to handle. The predictions are returned instead of the loss (in contrast with the other step
        functions).
        """
        preds, _, _ = self._generic_step(batch, batch_idx, self.pred_evaluator)
        return preds  # NOTE: this might be pretty intense on CPU/GPU memory for big data loaders!

    def on_predict_epoch_end(self, predictions):
        """Completes the epoch by asking the evaluator to summarize and log/dump its results.

        Note: for deployment-time predictions, there would be nothing to do here, but we abuse
        the pytorch lightning design a little bit here, and instead use this function to dump/log
        the final evaluation metrics for various data loaders. See the main.py/predict.py files
        where the `predict` functionality of the pytorch lightning module is called; in those
        cases, the `predict_eval_output_path` attribute of the model is set, and this is the
        trigger that activates the logging/dumping of the predictions.
        """
        # if the output path was not specified, there is nothing more to do here
        if self.predict_eval_output_path is not None \
                and not isinstance(self.pred_evaluator, eval_base.NoneEvaluator):
            # if the above attribute is set, then we'll actually dump the evaluator state and log!
            self.pred_evaluator.dump(self.predict_eval_output_path)
            # bonus: if the mlflow logger is available, it'll take responsibility of the dump
            if hasattr(self, "_mlf_logger") and os.path.exists(self.predict_eval_output_path):
                # note: the 'exists' catch above is to make sure we actually have something to save
                run_id = self._mlf_logger.run_id
                self._mlf_logger.experiment.log_artifact(
                    run_id=run_id,
                    local_path=self.predict_eval_output_path,
                    artifact_path="data_dumps",
                )
                # no need to keep two copies around --- delete the original, keep the logged one
                os.remove(self.predict_eval_output_path)
            # now, time to log the actual metrics to sigopt (if that's also available)
            prefix = pred_utils.get_eval_prefix_based_on_data_dump_path(
                self.predict_eval_output_path
            )
            if prefix is not None:
                results = self._get_latest_metric_eval_results(
                    prefix=prefix,
                    losses=None,
                    evaluator=self.pred_evaluator,
                )
                for metric_name, metric_val in results.items():
                    sigopt.log_metric(metric_name, metric_val)
        # finally, let's reset the prediction output path, as there's never a reason to overwrite
        self.predict_eval_output_path = None

    def _get_named_dataloader(self, prefix: str):
        """Get named dataloader.

        Extract a specific dataloader (i.e., train, val or test) according to its prefix.
        Args:
            prefix (str): one of train, val or test.

        Returns:
            dataloader: the relevant dataloader
        """
        assert prefix == 'train' or prefix == 'val' or prefix == 'test', "Wrong prefix."

        attribute_name = f"{prefix}_dataloader"

        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and self.trainer.datamodule is not None
        ):
            data_loader = self._get_dataloader_from_datamodule(self.trainer.datamodule, attribute_name)
        elif hasattr(self, "datamodule") and self.datamodule is not None:
            data_loader = self._get_dataloader_from_datamodule(self.datamodule, attribute_name)
        elif hasattr(self, "trainer") and self.trainer is not None:
            # this case is relevant if we invoke trainer.fit(model, dataloader=...)
            data_loader = self._get_dataloader_from_trainer(self.trainer, attribute_name)
        else:
            data_loader = None

        assert data_loader is not None and isinstance(
            data_loader, torch.utils.data.DataLoader
        ), "training data loader is not available: cannot proceed."
        return data_loader

    @staticmethod
    def _get_dataloader_from_datamodule(datamodule, dataloader_name):
        """Get a dataloader from a datamodule.

        Args:
            datamodule (Datamodule):  a PL datamodule
            dataloader_name (str): the identifier for the dataloader.

        Returns:
            dataloader: the "dataloader_name" dataloader
        """
        # NOTE: when getting the dataloader from a datamodule, the attribute is a FUNCTION.
        # Hence the parenthesis at the end to actually get the dataloader.
        data_loader = getattr(datamodule, dataloader_name)()
        return data_loader

    @staticmethod
    def _get_dataloader_from_trainer(trainer, dataloader_name):
        """Get a dataloader from a trainer.

        Args:
            datamodule (trainer):  a PL trainer
            dataloader_name (str): the identifier for the dataloader.

        Returns:
            dataloader: the "dataloader_name" dataloader
        """
        # NOTE: when getting the dataloader from a trainer, the attribute is
        # some PL wrapper around the dataloader. No PARENTHESIS AT THE END.
        try:
            combined_loader = getattr(trainer, dataloader_name)
            data_loader = combined_loader.loaders
        except Exception:
            if dataloader_name == "val_dataloader":
                data_loaders = getattr(trainer, "val_dataloaders")
                assert isinstance(data_loaders, list) and len(data_loaders) == 1
                data_loader = data_loaders[0]
            else:
                raise
        assert isinstance(data_loader, torch.utils.data.DataLoader)
        return data_loader

    def _get_train_dataloader(self):
        """Returns the training dataloader."""
        return self._get_named_dataloader("train")

    def _get_val_dataloader(self):
        """Returns the validation dataloader."""
        return self._get_named_dataloader("val")

    def _get_test_dataloader(self):
        """Returns the test dataloader."""
        return self._get_named_dataloader("test")

    def _get_persistent_data_id(
        self,
        batch: typing.Dict,
        batch_idx: int,  # index of the batch itself inside the epoch
        data_sample_idx: int,  # index of the data sample inside the batch we want to log
    ) -> typing.Any:
        """Returns the full 'identifier' (or name) used to uniquely tag a data sample.

        The batch idx should correspond to the index of the batch the sample was found in. This is
        not a robust value, as shuffling might change which batch all samples end up in. The
        data sample index is the index of the targeted sample inside the current batch.
        """
        assert 0 <= data_sample_idx < batch["batch_size"], "should never happen?"
        # this is application-specific, and needs to be redefined in the derived class!
        # (the default below might lose the synch when shuffling samples, but it's not catastrophic)
        return batch_idx, data_sample_idx

    def _pick_data_ids_to_render_and_log(
        self,
        prefix: typing.AnyStr,
        data_loader: torch.utils.data.DataLoader,
    ) -> typing.List[typing.Any]:
        """Returns the list of data sample ids that should be rendered/displayed during each epoch.

        The first time this code is executed, the list of sample ids to render will be returned
        based on the expected total number of elements that will be loaded. Subsequent calls will
        return the translated ids directly, meaning this is robust to shuffling.
        """
        if not self.images_to_display:
            return []  # nothing to render/log
        if prefix not in self.data_ids_to_render_and_log:  # return the ID list as-is if it exists
            assert isinstance(data_loader, torch.utils.data.DataLoader), \
                "unexpected data loader wrapper (we need to get batch size/counts somehow)"
            batch_size = self._get_batch_size_from_data_loader(data_loader)
            max_total_sample_count = batch_size * len(data_loader)
            # note: we'll display at most 'self.images_to_display' images without guarantees
            display_count = min(self.images_to_display, max_total_sample_count)
            # first, assign raw idxs as the targets; the first pass over the dataset will convert
            # them to proper image identifiers (application-specific) so that we can find them again
            # even if the data loader shuffles its indices every epoch
            picked_idxs = np.sort(np.random.permutation(max_total_sample_count)[:display_count])
            # we convert the picked indices into batch and sample indices that can be used in steps
            data_ids = [(idx // batch_size, idx % batch_size) for idx in picked_idxs]
            # note: each step, the ids will be scanned and potentially replaced by persistent ones!
            # (see `_get_data_id` and `_render_data_samples_from_ids` for more info)
            self.data_ids_to_render_and_log[prefix] = data_ids
        return self.data_ids_to_render_and_log[prefix]

    def _render_and_log_data_samples_from_ids(
        self,
        batch: typing.Dict,
        batch_idx: int,
        raw_preds: typing.Any,
        prefix: typing.AnyStr,
    ):
        """Extracts and renders data samples from the current batch (if possible).

        This function relies on the picked data sample IDs that are generated at the beginning of
        the 1st epoch of training/validation/testing. If a match is found for any picked id in the
        current batch, we will update the id itself to a permanent one (to be robust to shuffling)
        and render the corresponding sample for mlflow/tensorboard/others.
        """
        if prefix not in self.data_ids_to_render_and_log or not self.data_ids_to_render_and_log[prefix]:
            return  # quick exit if we don't actually want to render/log any predictions
        # the first time we run this code, we'll translate batch-and-sample-idxs tuples into real ids
        curr_batch_size = batch["batch_size"]
        for data_sample_idx in range(curr_batch_size):
            orig_data_id = (batch_idx, data_sample_idx)  # original id = from 1st epoch, to update
            data_id = self._get_persistent_data_id(batch, batch_idx, data_sample_idx)
            match_idx = None
            if orig_data_id in self.data_ids_to_render_and_log[prefix]:
                # check for the preliminary id match and update it if possible
                match_idx = self.data_ids_to_render_and_log[prefix].index(orig_data_id)
                self.data_ids_to_render_and_log[prefix][match_idx] = data_id
            elif data_id in self.data_ids_to_render_and_log[prefix]:
                # check for the persistent id match otherwise
                match_idx = self.data_ids_to_render_and_log[prefix].index(data_id)
            # if we do find a name match, we'll do the rendering and logging
            if match_idx is not None:
                self._render_and_log_data_sample(
                    batch=batch,
                    batch_idx=batch_idx,
                    data_sample_idx=data_sample_idx,
                    data_id=data_id,
                    raw_preds=raw_preds,
                    prefix=prefix,
                )

    def _render_and_log_data_sample(
        self,
        batch: typing.Dict,
        batch_idx: int,  # index of the batch itself inside the epoch
        data_sample_idx: int,  # index of the data sample inside the batch we want to log
        data_id: typing.Any,  # the hashable identifier used to name this particular sample
        raw_preds: typing.Any,
        prefix: typing.AnyStr,
    ) -> None:
        """Renders and logs a specific data sample from the current batch via available loggers."""
        # note: by default, we've got no idea what to render and how, so this does nothing
        return
