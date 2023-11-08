'''PINA Callbacks Implementations'''

from pytorch_lightning.callbacks import Callback
import torch
import copy


class MetricTracker(Callback):
    
    def __init__(self):
        """
        PINA Implementation of a Lightning Callback for Metric Tracking.

        This class provides functionality to track relevant metrics during the training process.

        :ivar _collection: A list to store collected metrics after each training epoch.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer

        :return: A dictionary containing aggregated metric values.
        :rtype: dict

        Example:
            >>> tracker = MetricTracker()
            >>> # ... Perform training ...
            >>> metrics = tracker.metrics
        """
        self._collection = []

    def on_train_epoch_end(self, trainer, __):
        """
        Collect and track metrics at the end of each training epoch.

        :param trainer: The trainer object managing the training process.
        :type trainer: pytorch_lightning.Trainer
        :param _: Placeholder argument.

        :return: None
        :rtype: None
        """
        self._collection.append(copy.deepcopy(
            trainer.logged_metrics))  # track them

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
