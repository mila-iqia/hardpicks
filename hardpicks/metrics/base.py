"""Base class module for all evaluator objects.

Also contains a 'None' evaluator that can be used to disable evaluation entirely.
"""

import typing


class EvaluatorBase:
    """Evaluator base with interface definitions.

    This class shows the task-agnostic capabilities of all evaluators. All functions below must
    be overloaded in derived classes as they will all be called in the PyTorch-Lightning modules.
    Derived evaluators are however free to interpret the data passed into the ``ingest'' function
    as necessary for a particular task (e.g. first break picking or seismic volume segmentation).
    """

    def ingest(
        self,
        batch: typing.Any,
        batch_idx: int,
        preds: typing.Any,
    ) -> typing.Dict[typing.AnyStr, float]:
        """Ingests the batch data and prediction from the model's step function.

        Each time this function is called, it will record whatever artifacts are necessary to
        compute evaluation summaries at the end of the epoch later (see ``summarize'').

        Arguments:
            batch: the minibatch provided by the data loader that contains all inputs/targets.
            batch_idx: the index of this particular step (or batch) in the current epoch.
            preds: the predictions made by the model based on the minibatch input data.

        Returns:
            A dictionary of metric evaluation results (if any 'live' results were requested).
        """
        raise NotImplementedError

    def summarize(
        self,
        category_name: typing.Optional[typing.AnyStr] = None,
    ) -> typing.Dict[typing.AnyStr, float]:
        """Returns the metric evaluation summary for all recorded prediction results.

        By default, if no category name is provided, the returned metrics will summarize all
        the prediction results. Otherwise, the metrics will be tied to a specific category.

        Arguments:
            category_name: the name of the evaluation category to summarize. Defaults to
                ``None'' meaning that all categories should be merged for an overall summary.

        Returns:
            A dictionary of metric evaluation results.
        """
        raise NotImplementedError

    def get_categories(self) -> typing.Sequence[typing.AnyStr]:
        """Returns the categories supported by this evaluator.

        These might have been requested through the hyperparameter configuration dictionary or
        hard-coded as part of the task that the implementation evaluates.

        Should be an empty list if the evaluator supports no categories.
        """
        raise NotImplementedError

    def reset(self):
        """Resets the internal state of the evaluator object (useful when starting a new epoch)."""
        raise NotImplementedError

    def dump(self, path: typing.AnyStr):
        """Dumps the internal state of the evaluator object at the given location for reinstantiation."""
        raise NotImplementedError

    @staticmethod
    def load(path: typing.AnyStr) -> "EvaluatorBase":
        """Reinstantiates an evaluator as previously dumped at the given location."""
        raise NotImplementedError

    def finalize(self):
        """Perform needed book-keeping at the end of a run, either an epoch, or an early-stopped epoch."""
        raise NotImplementedError


class NoneEvaluator(EvaluatorBase):
    """Dummy (pass-through) evaluator.

    This class shows the task-agnostic capabilities of all evaluators. All functions below must
    be overloaded in derived classes as they will all be called in the PyTorch-Lightning modules.
    Derived evaluators are however free to interpret the data passed into the ``ingest'' function
    as necessary for a particular task (e.g. first break picking or seismic volume segmentation).
    """
    def __init__(
        self,
        hyper_params: typing.Any,
    ):
        """Pass through dummy init."""
        pass

    def ingest(
        self,
        batch: typing.Any,
        batch_idx: int,
        preds: typing.Any,
    ) -> typing.Dict[typing.AnyStr, float]:
        """Ingests the batch data and prediction, but does nothing."""
        return dict()

    def summarize(
        self,
        category_name: typing.Optional[typing.AnyStr] = None,
    ) -> typing.Dict[typing.AnyStr, float]:
        """Returns the (empty) metric evaluation summary."""
        return dict()

    def get_categories(self) -> typing.Sequence[typing.AnyStr]:
        """Returns an empty list (this evaluator has no categories)."""
        return list()

    def reset(self):
        """Does nothing (this evaluator has nothing to reset between epochs)."""
        return

    def dump(self, path: typing.AnyStr):
        """Dumps nothing to disk (there's nothing to restore)."""
        return

    @staticmethod
    def load(path: typing.AnyStr) -> "NoneEvaluator":
        """Loads nothing from disk (there's nothing to restore)."""
        return NoneEvaluator(hyper_params=None)

    def finalize(self):
        """Does nothing (no need to finalize ingested data)."""
        return
