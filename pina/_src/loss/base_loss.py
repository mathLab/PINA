"""Module for the BaseLoss class."""

import torch
from pina._src.loss.loss_interface import LossInterface


class BaseLoss(LossInterface):
    """
    Base class for all losses, implementing common functionality.

    All specific loss types should inherit from this class and implement its
    abstract methods.

    This class is not meant to be instantiated directly.
    """

    # Define available reduction methods
    _REDUCTION_METHOD = {
        "sum": lambda x: torch.sum(x, keepdim=True, dim=-1),
        "mean": lambda x: torch.mean(x, keepdim=True, dim=-1),
        "none": lambda x: x,
    }

    def __init__(self, reduction="mean"):
        """
        Initialization of the :class:`BaseLoss` class.

        :param str reduction: The reduction method to aggregate pointwise loss
            values. Available options include: ``"none"`` for unreduced loss,
            ``"mean"`` for the average of the loss values, and ``"sum"`` for
            their total sum. Default is ``"mean"``.
        :raises ValueError: If the specified reduction method is not among the
            available options.
        """
        # Check that the reduction method is available
        if reduction not in self._REDUCTION_METHOD:
            raise ValueError(
                f"Invalid reduction method. Available options: "
                f"{list(self._REDUCTION_METHOD.keys())}. Got {reduction}."
            )

        # Initialization
        super().__init__(reduction=reduction, size_average=None, reduce=None)

    def _reduction(self, loss):
        """
        Apply the configured reduction operation to pointwise loss values.

        :param torch.Tensor loss: The tensor of pointwise losses.
        :return: The reduced loss tensor.
        :rtype: torch.Tensor
        """
        return self._REDUCTION_METHOD[self.reduction](loss)
