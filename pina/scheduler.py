""" Module for PINA Scheduler """

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0
from .optimizer import Optimizer
from .utils import check_consistency


class Scheduler: # TODO improve interface
    pass


class TorchScheduler(Scheduler):

    def __init__(self, scheduler_class, **kwargs):
        check_consistency(scheduler_class, LRScheduler, subclass=True)

        self.scheduler_class = scheduler_class
        self.kwargs = kwargs

    def hook(self, optimizer):
        check_consistency(optimizer, Optimizer)
        self.scheduler_instance = self.scheduler_class(
            optimizer.optimizer_instance, **self.kwargs
        )
