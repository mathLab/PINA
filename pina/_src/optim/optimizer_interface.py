"""Module for the Optimizer Interface."""

from abc import ABCMeta, abstractmethod


class OptimizerInterface(metaclass=ABCMeta):
    """
    Abstract interface for all optimizers.

    :Example:

        >>> from pina.optim import TorchOptimizer
        >>> import torch
        >>> opt = TorchOptimizer(torch.optim.Adam, lr=0.01)
        >>> isinstance(opt, OptimizerInterface)
        True
    """

    @abstractmethod
    def hook(self, parameters):
        """
        Execute custom logic associated with the optimizer instance.

        This method is intended to encapsulate any additional behavior that
        should be triggered during the optimization process.

        :param dict parameters: The parameters of the model to be optimized.
        """

    @property
    @abstractmethod
    def instance(self):
        """
        The underlying optimizer object.

        :return: The optimizer instance.
        :rtype: object
        """
