"""Module for wrapping PyTorch optimizers."""

import torch
from pina._src.core.utils import check_consistency
from pina._src.optim.optimizer_interface import OptimizerInterface


class TorchOptimizer(OptimizerInterface):
    """
    The wrapper class for PyTorch optimizers.

    This class wraps a ``torch.optim.Optimizer`` class and defers its
    instantiation until runtime. It enables a consistent interface across
    different optimizer backends while leveraging PyTorch's optimization
    algorithms.

    :Example:

        >>> from pina.optim import TorchOptimizer
        >>> import torch
        >>> optimizer = TorchOptimizer(torch.optim.Adam, lr=0.001)
        >>> optimizer.optimizer_class
        <class 'torch.optim.adam.Adam'>
    """

    def __init__(self, optimizer_class, **kwargs):
        """
        Initialization of the :class:`TorchOptimizer` class.

        :param torch.optim.Optimizer optimizer_class: The subclass of
            ``torch.optim.Optimizer`` to be instantiated.
        :param dict kwargs: Additional keyword arguments forwarded to the
            optimizer constructor. See more
            `here <https://pytorch.org/docs/stable/optim.html#algorithms>`_.
        :raises ValueError: If ``optimizer_class`` is not a subclass of
            ``torch.optim.Optimizer``.
        """
        # Check consistency
        check_consistency(optimizer_class, torch.optim.Optimizer, subclass=True)

        # Initialize attributes
        self.optimizer_class = optimizer_class
        self.kwargs = kwargs
        self._optimizer_instance = None

    def hook(self, parameters):
        """
        Execute custom logic associated with the optimizer instance.

        This method is intended to encapsulate any additional behavior that
        should be triggered during the optimization process.

        :param dict parameters: The parameters of the model to be optimized.
        """
        self._optimizer_instance = self.optimizer_class(
            parameters, **self.kwargs
        )

    @property
    def instance(self):
        """
        The underlying optimizer object.

        :return: The optimizer instance.
        :rtype: torch.optim.Optimizer
        """
        return self._optimizer_instance
