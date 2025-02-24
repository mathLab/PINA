"""Module for Loss Interface"""

from abc import ABCMeta, abstractmethod


class WeightingInterface(metaclass=ABCMeta):
    """
    The ``weightingInterface`` class. TODO
    """

    def __init__(self):
        self.condition_names = None

    @abstractmethod
    def aggregate(self, losses):
        """
        Aggregate the losses.

        :param dict(torch.Tensor) input: The dictionary of losses.
        :return: The losses aggregation. It should be a scalar Tensor.
        :rtype: torch.Tensor
        """
        pass
