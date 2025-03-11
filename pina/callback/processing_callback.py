"""PINA Callbacks Implementations"""

import copy
import torch

from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from lightning.pytorch.callbacks.progress.progress_bar import (
    get_standard_metrics,
)
from pina.utils import check_consistency


class MetricTracker(Callback):
    """
    Lightning Callback for Metric Tracking.
    """

    def __init__(self, metrics_to_track=None):
        """
        Tracks specified metrics during training.

        :param metrics_to_track: List of metrics to track.
            Defaults to train loss.
        :type metrics_to_track: list, optional
        """
        super().__init__()
        self._collection = []
        # Default to tracking 'train_loss' if not specified
        self.metrics_to_track = metrics_to_track

    def setup(self, trainer, pl_module, stage):
        """
        Called when fit, validate, test, predict, or tune begins.

        :param Trainer trainer: A :class:`~pina.trainer.Trainer` instance.
        :param SolverInterface pl_module: A
            :class:`~pina.solver.solver.SolverInterface` instance.
        :param str stage: Either 'fit', 'test' or 'predict'.
        """
        if self.metrics_to_track is None and trainer.batch_size is None:
            self.metrics_to_track = ["train_loss"]
        elif self.metrics_to_track is None:
            self.metrics_to_track = ["train_loss_epoch"]
        return super().setup(trainer, pl_module, stage)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Collect and track metrics at the end of each training epoch.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param pl_module: The model being trained (not used here).
        """
        # Track metrics after the first epoch onwards
        if trainer.current_epoch > 0:
            # Append only the tracked metrics to avoid unnecessary data
            tracked_metrics = {
                k: v
                for k, v in trainer.logged_metrics.items()
                if k in self.metrics_to_track
            }
            self._collection.append(copy.deepcopy(tracked_metrics))

    @property
    def metrics(self):
        """
        Aggregate collected metrics over all epochs.

        :return: A dictionary containing aggregated metric values.
        :rtype: dict
        """
        if not self._collection:
            return {}

        # Get intersection of keys across all collected dictionaries
        common_keys = set(self._collection[0]).intersection(
            *self._collection[1:]
        )

        # Stack the metric values for common keys and return
        return {
            k: torch.stack([dic[k] for dic in self._collection])
            for k in common_keys
            if k in self.metrics_to_track
        }


class PINAProgressBar(TQDMProgressBar):
    """
    PINA Implementation of a Lightning Callback for enriching the progress bar.
    """

    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
        "{rate_noinv_fmt}{postfix}]"
    )

    def __init__(self, metrics="val", **kwargs):
        """
        This class enables the display of only relevant metrics during training.

        :param metrics: Logged metrics to be shown during the training.
            Must be a subset of the conditions keys defined in
            :obj:`pina.condition.Condition`.
        :type metrics: str | list(str) | tuple(str)

        :Keyword Arguments:
            The additional keyword arguments specify the progress bar and can be
            choosen from the `pytorch-lightning TQDMProgressBar API
            <https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callback/progress/tqdm_progress.html#TQDMProgressBar>`_

        Example:
            >>> pbar = PINAProgressBar(['mean'])
            >>> # ... Perform training ...
            >>> trainer = Trainer(solver, callbacks=[pbar])
        """
        super().__init__(**kwargs)
        # check consistency
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        check_consistency(metrics, str)
        self._sorted_metrics = metrics

    def get_metrics(self, trainer, pl_module):
        r"""Combine progress bar metrics collected from the trainer with
        standard metrics from get_standard_metrics.
        Override this method to customize the items shown in the progress bar.
        The progress bar metrics are sorted according to ``metrics``.

        Here is an example of how to override the defaults:

        .. code-block:: python

            def get_metrics(self, trainer, model):
                # don't show the version number
                items = super().get_metrics(trainer, model)
                items.pop("v_num", None)
                return items

        :return: Dictionary with the items to be displayed in the progress bar.
        :rtype: tuple(dict)
        """
        standard_metrics = get_standard_metrics(trainer)
        pbar_metrics = trainer.progress_bar_metrics
        if pbar_metrics:
            pbar_metrics = {
                key: pbar_metrics[key]
                for key in pbar_metrics
                if key in self._sorted_metrics
            }
        return {**standard_metrics, **pbar_metrics}

    def setup(self, trainer, pl_module, stage):
        """
        Check that the initialized metrics are available and correctly logged.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param pl_module: Placeholder argument.
        """
        # Check if all keys in sort_keys are present in the dictionary
        for key in self._sorted_metrics:
            if (
                key not in trainer.solver.problem.conditions.keys()
                and key != "train"
                and key != "val"
            ):
                raise KeyError(f"Key '{key}' is not present in the dictionary")
        # add the loss pedix
        if trainer.batch_size is not None:
            pedix = "_loss_epoch"
        else:
            pedix = "_loss"
        self._sorted_metrics = [
            metric + pedix for metric in self._sorted_metrics
        ]
        return super().setup(trainer, pl_module, stage)
