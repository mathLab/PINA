"""Module for the Time Weighting Interface."""

from abc import ABCMeta, abstractmethod
import torch


class TimeWeightingInterface(metaclass=ABCMeta):
    """
    Abstract base class for all time weighting schemas. All time weighting
    schemas should inherit from this class.
    """

    @abstractmethod
    def __call__(self, num_steps, device):
        """
        Compute the weights for the time steps.

        :param int num_steps: The number of time steps.
        :param torch.device device: The device on which the weights should be
            created.
        :return: The weights for the time steps.
        :rtype: torch.Tensor
        """
        pass
