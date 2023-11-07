'''PINA Callbacks Implementations'''

from pytorch_lightning.callbacks import Callback
import torch
import copy


class MetricTracker(Callback):
    """
    PINA implementation of a Lightining Callback to track relevant
    metrics during training.
    """
    def __init__(self):
        self._collection = []

    def on_train_epoch_end(self, trainer, __):
        self._collection.append(copy.deepcopy(trainer.logged_metrics)) # track them

    @property
    def metrics(self):
        common_keys = set.intersection(*map(set, self._collection))
        v = {k: torch.stack([dic[k] for dic in self._collection]) for k in common_keys}
        return v

    