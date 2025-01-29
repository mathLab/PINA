""" Module for Loss Interface """

from abc import ABCMeta, abstractmethod


class weightningInterface(metaclass=ABCMeta):
    """
    The ``weightingInterface`` class. TODO
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def aggregate(self, losses):
        """
        Aggregate the losses.

        :param list(torch.Tensor) input: The list
        :return: The losses aggregation. It should be a scalar Tensor.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def NTK_weighting(self, losses):
        """
        Weight the losses 
        """
        pass