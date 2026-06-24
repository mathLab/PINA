"""Module for the Metric Tracker."""

import copy
import torch
from lightning.pytorch.callbacks import Callback
from pina._src.core.utils import check_consistency


class MetricTracker(Callback):
    """
    Callback for collecting selected metrics logged during training.

    :Example:

        >>> tracker = MetricTracker(metrics=["train_loss"])
        >>> tracker.metrics_to_track
        ['train_loss']
    """

    def __init__(self, metrics_to_track=None):
        """
        Initialization of the :class:`MetricTracker` class.

        :param metrics_to_track: The names of the metrics to collect. If
            ``None``, defaults to ``["train_loss"]`` when no batch size is
            available, otherwise to ``["train_loss_epoch"]``. Default is
            ``None``.
        :type metrics_to_track: str | list[str]
        :raises ValueError: If any of the provided metric names are not strings.
        """
        super().__init__()

        # Check consistency
        if metrics_to_track is not None:
            check_consistency(metrics_to_track, str)

        # Convert to list if a single string is provided
        if isinstance(metrics_to_track, str):
            metrics_to_track = [metrics_to_track]

        # Initialize the collection list and store the metrics to track
        self.metrics_to_track = metrics_to_track
        self._collection = []

    def setup(self, trainer, pl_module, stage):
        """
        Configure the metrics to track before execution starts.

        When a batch size is provided (i.e. ``trainer.batch_size`` is not
        ``None``), metric names are expanded to match Lightning's logging
        convention: for each metric ``m``, both ``m_step`` and ``m_epoch`` are
        tracked. For example, ``"train_loss"`` becomes
        ``["train_loss_step", "train_loss_epoch"]``.

        :param Trainer trainer: The trainer instance managing the execution.
        :param BaseSolver pl_module: The solver module being executed.
        :param str stage: Current execution stage.
        """
        # Set default metrics to train_loss if no batch size is available
        if self.metrics_to_track is None:
            self.metrics_to_track = ["train_loss"]

        # If a batch size is provided, expand metric names to match convention
        if trainer.batch_size is not None:
            self.metrics_to_track = [
                f"{metric}_{suffix}"
                for metric in self.metrics_to_track
                for suffix in ("step", "epoch")
            ]

        return super().setup(trainer, pl_module, stage)

    def on_train_epoch_end(self, trainer, __):
        """
        Store the selected logged metrics at the end of each training epoch.

        :param Trainer trainer: The trainer instance managing the execution.
        :param __: Placeholder argument, not used.
        """
        # Only collect metrics after the first epoch to ensure they are logged
        if trainer.current_epoch > 0:

            # Collect the metrics that are being tracked
            tracked_metrics = {
                k: v
                for k, v in trainer.logged_metrics.items()
                if k in self.metrics_to_track
            }
            self._collection.append(copy.deepcopy(tracked_metrics))

    @property
    def metrics(self):
        """
        Return the collected metrics stacked over the tracked epochs.

        :return: The dictionary mapping each metric name to a tensor containing
            its values across epochs. Returns an empty dictionary if no metrics
            have been collected.
        :rtype: dict[str, torch.Tensor]
        """
        if not self._collection:
            return {}

        # Identify the common keys across all collected metric dictionaries
        common_keys = set(self._collection[0]).intersection(
            *self._collection[1:]
        )

        return {
            k: torch.stack([dic[k] for dic in self._collection])
            for k in common_keys
            if k in self.metrics_to_track
        }
