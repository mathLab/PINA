"""Module for wrapping PyTorch schedulers."""

from torch.optim.lr_scheduler import LRScheduler
from pina._src.core.utils import check_consistency
from pina._src.optim.optimizer_interface import OptimizerInterface
from pina._src.optim.scheduler_interface import SchedulerInterface


class TorchScheduler(SchedulerInterface):
    """
    The wrapper class for PyTorch schedulers.

    This class wraps a ``torch.optim.lr_scheduler.LRScheduler`` class and defers
    its instantiation until runtime, once the optimizer instance is available.

    :Example:

        >>> from pina.optim import TorchScheduler
        >>> import torch
        >>> scheduler = TorchScheduler(
        ...     torch.optim.lr_scheduler.StepLR, step_size=5)
        >>> scheduler.scheduler_class
        <class 'torch.optim.lr_scheduler.StepLR'>
    """

    def __init__(self, scheduler_class, **kwargs):
        """
        Initialization of the :class:`TorchScheduler` class.

        :param torch.optim.LRScheduler scheduler_class: The subclass of
            ``torch.optim.lr_scheduler.LRScheduler`` to be instantiated.
        :param dict kwargs: Additional keyword arguments forwarded to the
            scheduler constructor. See more
            `here <https://pytorch.org/docs/stable/optim.html#algorithms>`_.
        :raises ValueError: If ``scheduler_class`` is not a subclass of
            ``torch.optim.lr_scheduler.LRScheduler``.
        """
        # Check consistency
        check_consistency(scheduler_class, LRScheduler, subclass=True)

        # Initialize attributes
        self.scheduler_class = scheduler_class
        self.kwargs = kwargs
        self._scheduler_instance = None

    def hook(self, optimizer):
        """
        Initialize the scheduler instance with the given parameters.

        :param OptimizerInterface optimizer: The optimizer instance associated
            with the scheduler.
        :raises ValueError: If ``optimizer`` is not an instance of
            :class:`OptimizerInterface`.
        """
        # Check consistency
        check_consistency(optimizer, OptimizerInterface)

        # Initialize the scheduler instance
        self._scheduler_instance = self.scheduler_class(
            optimizer.instance, **self.kwargs
        )

    @property
    def instance(self):
        """
        The underlying scheduler object.

        :return: The scheduler instance.
        :rtype: torch.optim.lr_scheduler.LRScheduler
        """
        return self._scheduler_instance
