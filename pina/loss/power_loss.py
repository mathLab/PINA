"""Module for PowerLoss class"""

import torch

from ..utils import check_consistency
from .loss_interface import LossInterface


class PowerLoss(LossInterface):
    r"""
    The PowerLoss loss implementation class. Creates a criterion that measures
    the error between each element in the input :math:`x` and
    target :math:`y` powered to a specific integer.

    The unreduced (i.e. with ``reduction`` set to ``none``) loss can
    be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{1}{D}\left[\sum_{i=1}^{D} 
        \left| x_n^i - y_n^i \right|^p\right],
    
    If ``'relative'`` is set to true:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{ \sum_{i=1}^{D} | x_n^i - y_n^i|^p }
        {\sum_{i=1}^{D}|y_n^i|^p},

    where :math:`N` is the batch size. If ``reduction`` is not ``none``
    (default ``mean``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by 
    :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction`` to 
    ``sum``.
    """

    def __init__(self, p=2, reduction="mean", relative=False):
        """
        :param int p: Degree of Lp norm. It specifies the type of norm to
            be calculated. See `list of possible orders in torch linalg
            <https://pytorch.org/docs/stable/generated/
            torch.linalg.norm.html#torch.linalg.norm>`_ to
            see the possible degrees. Default 2 (euclidean norm).
        :param str reduction: Specifies the reduction to apply to the output:
            ``none`` | ``mean`` | ``sum``. When ``none``: no reduction
            will be applied, ``mean``: the sum of the output will be divided
            by the number of elements in the output, ``sum``: the output will
            be summed.
        :param bool relative: Specifies if relative error should be computed.
        """
        super().__init__(reduction=reduction)

        # check consistency
        check_consistency(p, (str, int, float))
        check_consistency(relative, bool)

        self.p = p
        self.relative = relative

    def forward(self, input, target):
        """Forward method for loss function.

        :param torch.Tensor input: Input tensor from real data.
        :param torch.Tensor target: Model tensor output.
        :return: Loss evaluation.
        :rtype: torch.Tensor
        """
        loss = torch.abs((input - target)).pow(self.p).mean(-1)
        if self.relative:
            loss = loss / torch.abs(input).pow(self.p).mean(-1)
        return self._reduction(loss)
