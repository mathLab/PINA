"""Module for the PINA Torch Optimizer"""

import copy

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from ..utils import check_consistency
from .core.scheduler_connector import SchedulerConnector


class TorchScheduler(SchedulerConnector):
    """
    A wrapper class for using PyTorch schedulers.
    """

    def __init__(self, scheduler_class, **scheduler_kwargs):
        """
        Initialization of the :class:`TorchScheduler` class.

        :param torch.optim.LRScheduler scheduler_class: A
            :class:`torch.optim.LRScheduler` class.
        :param dict kwargs: Additional parameters passed to ``scheduler_class``,
            see more
            `here <https://pytorch.org/docs/stable/optim.html#algorithms>_`.
        """
        check_consistency(scheduler_class, LRScheduler, subclass=True)
        check_consistency(scheduler_kwargs, dict)
        super().__init__(scheduler_class, **scheduler_kwargs)
