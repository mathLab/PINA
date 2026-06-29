"""Module for the Optimizer Interface."""

from abc import ABCMeta, abstractmethod


class OptimizerInterface(metaclass=ABCMeta):
    """
    Abstract interface for all optimizers.
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
