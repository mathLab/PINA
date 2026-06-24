"""Module for the Loss Interface."""

from abc import ABCMeta, abstractmethod
from torch.nn.modules.loss import _Loss


class DualLossInterface(_Loss, metaclass=ABCMeta):
    """
    Abstract interface for all losses requiring both an input and a target
    tensor.

    :Example:

        >>> import torch
        >>> from pina.loss import LpLoss
        >>> loss = LpLoss(p=2)
        >>> input = torch.randn(10, 3)
        >>> target = torch.randn(10, 3)
        >>> loss(input, target)
        tensor(...)
    """

    @abstractmethod
    def forward(self, input, target):
        """
        Forward method of the loss function.

        :param torch.Tensor input: The input tensor.
        :param torch.Tensor target: The target tensor.
        :return: The computed loss.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def _reduction(self, loss):
        """
        Apply the configured reduction operation to pointwise loss values.

        :param torch.Tensor loss: The tensor of pointwise losses.
        :return: The reduced loss tensor.
        :rtype: torch.Tensor
        """
