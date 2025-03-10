"""Module for Loss Interface"""

from abc import ABCMeta, abstractmethod
from torch.nn.modules.loss import _Loss
import torch


class LossInterface(_Loss, metaclass=ABCMeta):
    """
    The abstract ``LossInterface`` class. All the class defining a PINA Loss
    should be inheritied from this class.
    """

    def __init__(self, reduction="mean"):
        """
        :param str reduction: Specifies the reduction to apply to the output:
            ``none`` | ``mean`` | ``sum``. When ``none``: no reduction
            will be applied, ``mean``: the sum of the output will be divided
            by the number of elements in the output, ``sum``: the output will
            be summed. Note: ``size_average`` and ``reduce`` are in the
            process of being deprecated, and in the meantime, specifying either
            of those two args will override ``reduction``. Default: ``mean``.
        """
        super().__init__(reduction=reduction, size_average=None, reduce=None)

    @abstractmethod
    def forward(self, input, target):
        """Forward method for loss function.

        :param torch.Tensor input: Input tensor from real data.
        :param torch.Tensor target: Model tensor output.
        :return: Loss evaluation.
        :rtype: torch.Tensor
        """

    def _reduction(self, loss):
        """Simple helper function to check reduction

        :param reduction: Specifies the reduction to apply to the output:
            ``none`` | ``mean`` | ``sum``. When ``none``: no reduction
            will be applied, ``mean``: the sum of the output will be divided
            by the number of elements in the output, ``sum``: the output will
            be summed. Note: ``size_average`` and ``reduce`` are in the
            process of being deprecated, and in the meantime, specifying either
            of those two args will override ``reduction``. Default: ``mean``.
        :type reduction: str
        :param loss: Loss tensor for each element.
        :type loss: torch.Tensor
        :return: Reduced loss.
        :rtype: torch.Tensor
        """
        if self.reduction == "none":
            ret = loss
        elif self.reduction == "mean":
            ret = torch.mean(loss, keepdim=True, dim=-1)
        elif self.reduction == "sum":
            ret = torch.sum(loss, keepdim=True, dim=-1)
        else:
            raise ValueError(self.reduction + " is not valid")
        return ret
