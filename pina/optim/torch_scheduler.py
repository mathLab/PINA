"""Module for PINA Torch Optimizer"""

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from ..utils import check_consistency
from .optimizer_interface import Optimizer
from .scheduler_interface import Scheduler


class TorchScheduler(Scheduler):
    """
    TODO

    :param Scheduler: _description_
    :type Scheduler: _type_
    """

    def __init__(self, scheduler_class, **kwargs):
        """
        TODO

        :param scheduler_class: _description_
        :type scheduler_class: _type_
        """
        check_consistency(scheduler_class, LRScheduler, subclass=True)

        self.scheduler_class = scheduler_class
        self.kwargs = kwargs
        self._scheduler_instance = None

    def hook(self, optimizer):
        """
        TODO

        :param optimizer: _description_
        :type optimizer: _type_
        """
        check_consistency(optimizer, Optimizer)
        self._scheduler_instance = self.scheduler_class(
            optimizer.instance, **self.kwargs
        )

    @property
    def instance(self):
        """
        Scheduler instance.
        """
        return self._scheduler_instance
