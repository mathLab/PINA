"""Module for the Lp Loss class."""

import torch
from pina._src.loss.base_loss import BaseLoss
from pina._src.core.utils import check_consistency


class LpLoss(BaseLoss):
    r"""
    Implementation of the :math:`L^p` loss measuring the pointwise :math:`L^p`
    distance between an input tensor :math:`x` and a target tensor :math:`y`.

    Given a batch of size :math:`N` and feature dimension :math:`D`, the
    unreduced loss (``reduction="none"``) is defined as:

    .. math::
        L = \{l_1, \dots, l_N\}^\top, \quad
        l_n = \left( \sum_{i=1}^{D} \left| x_n^i - y_n^i \right|^p \right)^{1/p}

    If ``relative=True``, each term is normalized by the :math:`L^p` norm of the
    input tensor :math:`x`:

    .. math::
        l_n = \frac{\left( \sum_{i=1}^{D} |x_n^i - y_n^i|^p \right)^{1/p}}
                {\left( \sum_{i=1}^{D} |x_n^i|^p \right)^{1/p}}

    If ``reduction`` is set to ``"mean"`` or ``"sum"``, the vector :math:`L`
    is aggregated accordingly:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{``mean''} \\
            \operatorname{sum}(L),  & \text{if reduction} = \text{``sum''}
        \end{cases}

    where :math:`N` is the batch size.
    """

    def __init__(self, p=2, reduction="mean", relative=False):
        """
        Initialization of the :class:`LpLoss` class.

        :param p: The order of the norm. It can be a numeric value for standard
            p-norms or one of the following strings: ``"inf"`` for maximum
            absolute value, ``"-inf"`` for minimum absolute value. The values
            ``"inf"`` and ``"-inf"`` are internally converted to their floating
            counterparts. Default is ``2``.
        :type p: int | float | str
        :param str reduction: The reduction method to aggregate pointwise loss
            values. Available options include: ``"none"`` for unreduced loss,
            ``"mean"`` for the average of the loss values, and ``"sum"`` for
            their total sum. Default is ``"mean"``.
        :param bool relative: If ``True``, computes the relative error.
            Default is ``False``.
        :raises ValueError: If ``relative`` is not a boolean.
        :raises ValueError: If ``p`` is not a valid norm order.
        """
        super().__init__(reduction=reduction)

        # Convert to float if inf or -inf
        if p == "inf":
            p = float("inf")
        elif p == "-inf":
            p = float("-inf")

        # Check consistency
        check_consistency(relative, bool)
        check_consistency(p, (int, float))

        # Initialize attributes
        self.p = p
        self.relative = relative

    def forward(self, input, target):
        """
        Forward method of the loss function.

        :param torch.Tensor input: The input tensor.
        :param torch.Tensor target: The target tensor.
        :return: The computed loss.
        :rtype: torch.Tensor
        """
        # Compute the standard loss
        loss = torch.linalg.norm((input - target), ord=self.p, dim=-1)

        # Compute the input norm for relative error
        if self.relative:
            loss = loss / torch.linalg.norm(input, ord=self.p, dim=-1)

        return self._reduction(loss)
