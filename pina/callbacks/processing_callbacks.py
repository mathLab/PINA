"""PINA Callbacks Implementations"""

from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.trainer.trainer import Trainer
import torch
import copy

from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from lightning.pytorch.callbacks.progress.progress_bar import (
    get_standard_metrics,
)
from pina.utils import check_consistency


class MetricTracker(Callback):

    def __init__(self, metrics_to_track=None):
        """
        PINA Implementation of a Lightning Callback for Metric Tracking.

        This class provides functionality to track relevant metrics during
        the training process.

        :ivar _collection: A list to store collected metrics after each
        training epoch.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer

        :return: A dictionary containing aggregated metric values.
        :rtype: dict

        Example:
            >>> tracker = MetricTracker()
            >>> # ... Perform training ...
            >>> metrics = tracker.metrics
        """
        super().__init__()
        self._collection = []
        if metrics_to_track is not None:
            metrics_to_track = ['train_loss_epoch', 'train_loss_step', 'val_loss']
        self.metrics_to_track = metrics_to_track

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Collect and track metrics at the end of each training epoch.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param pl_module: Placeholder argument.
        """
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.current_epoch > 0:
            self._collection.append(
                copy.deepcopy(trainer.logged_metrics)
            )  # track them

    @property
    def metrics(self):
        """
        Aggregate collected metrics during training.

        :return: A dictionary containing aggregated metric values.
        :rtype: dict
        """
        common_keys = set.intersection(*map(set, self._collection))
        v = {
            k: torch.stack([dic[k] for dic in self._collection])
            for k in common_keys
        }
        return v


class PINAProgressBar(TQDMProgressBar):

    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"

    def __init__(self, metrics="val_loss", **kwargs):
        """
        PINA Implementation of a Lightning Callback for enriching the progress
        bar.

        This class provides functionality to display only relevant metrics
        during the training process.

        :param metrics: Logged metrics to display during the training. It should
            be a subset of the conditions keys defined in
            :obj:`pina.condition.Condition`.
        :type metrics: str | list(str) | tuple(str)

        :Keyword Arguments:
            The additional keyword arguments specify the progress bar
            and can be choosen from the `pytorch-lightning
            TQDMProgressBar API <https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/progress/tqdm_progress.html#TQDMProgressBar>`_

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
        r"""Combines progress bar metrics collected from the trainer with
        standard metrics from get_standard_metrics.
        Implement this to override the items displayed in the progress bar.
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
                key: pbar_metrics[key] for key in self._sorted_metrics
            }
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
            rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                " If this is undesired, change the name or override `get_metrics()` in the progress bar callback.",
            )

        return {**standard_metrics, **pbar_metrics}

    def on_fit_start(self, trainer, pl_module):
        """
        Check that the metrics defined in the initialization are available,
        i.e. are correctly logged.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param pl_module: Placeholder argument.
        """
        # Check if all keys in sort_keys are present in the dictionary
        for key in self._sorted_metrics:
            if (
                key not in trainer.solver.problem.conditions.keys()
                and key != "mean"
            ):
                raise KeyError(f"Key '{key}' is not present in the dictionary")
        # add the loss pedix
        self._sorted_metrics = [
            metric + "_loss" for metric in self._sorted_metrics
        ]
        return super().on_fit_start(trainer, pl_module)
