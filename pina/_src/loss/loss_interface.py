"""Module for the Loss Interface."""

from abc import ABCMeta, abstractmethod
from torch.nn.modules.loss import _Loss


class DualLossInterface(_Loss, metaclass=ABCMeta):
    """
    Abstract interface for all losses.
    """

    def __init__(self, reduction="mean"):
        """
        Initialization of the :class:`DualLossInterface` class.

        :param str reduction: The reduction method for the loss.
            Available options: ``none``, ``mean``, ``sum``.
            If ``none``, no reduction is applied. If ``mean``, the sum of the
            loss values is divided by the number of values. If ``sum``, the loss
            values are summed. Default is ``mean``.
        """
        super().__init__(reduction=reduction, size_average=None, reduce=None)

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
