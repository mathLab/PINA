""" Module for Loss class """


from abc import ABCMeta, abstractmethod
from torch.nn.modules.loss import _Loss
import torch
from .utils import check_consistency

__all__ = ['LpLoss']

class LossInterface(_Loss, metaclass=ABCMeta):
    """
    The abstract `LossInterface` class. All the class defining a PINA Loss
    should be inheritied from this class.
    """

    def __init__(self, reduction = 'mean'):
        """
        :param str reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction 
            will be applied, ``'mean'``: the sum of the output will be divided
            by the number of elements in the output, ``'sum'``: the output will
            be summed. Note: :attr:`size_average` and :attr:`reduce` are in the
            process of being deprecated, and in the meantime, specifying either of
            those two args will override :attr:`reduction`. Default: ``'mean'``.
        """
        super().__init__(reduction=reduction, size_average=None, reduce=None)

    @abstractmethod
    def forward(self):
        pass

    def _reduction(self, loss):
        """Simple helper function to check reduction

        :param reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction 
            will be applied, ``'mean'``: the sum of the output will be divided
            by the number of elements in the output, ``'sum'``: the output will
            be summed. Note: :attr:`size_average` and :attr:`reduce` are in the
            process of being deprecated, and in the meantime, specifying either of
            those two args will override :attr:`reduction`. Default: ``'mean'``.
        :type reduction: str, optional
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

class LpLoss(LossInterface):
    r"""
    The Lp loss implementation class. Creates a criterion that measures
    the Lp error between each element in the input :math:`x` and
    target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``none``) loss can
    be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left[\sum_{i=1}^{D} \left| x_n^i - y_n^i \right|^p \right],
    
    If ``'relative'`` is set to true:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{ [\sum_{i=1}^{D} | x_n^i - y_n^i|^p] }{[\sum_{i=1}^{D}|y_n^i|^p]},

    where :math:`N` is the batch size. If :attr:`reduction` is not ``none``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets :attr:`reduction` to ``sum``.
    """

    def __init__(self, p=2, reduction = 'mean', relative = False):
        """
        :param int p: Degree of Lp norm. It specifies the type of norm to
            be calculated. See :meth:`torch.linalg.norm` ```'ord'``` to
            see the possible degrees. Default 2 (euclidean norm).
        :param str reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction 
            will be applied, ``'mean'``: the sum of the output will be divided
            by the number of elements in the output, ``'sum'``: the output will
            be summed. Note: :attr:`size_average` and :attr:`reduce` are in the
            process of being deprecated, and in the meantime, specifying either of
            those two args will override :attr:`reduction`. Default: ``'mean'``.
        :param bool relative: Specifies if relative error should be computed.
        """
        super().__init__(reduction=reduction)

        # check consistency
        check_consistency(p, (str,int,float))
        self.p = p
        check_consistency(relative, bool)
        self.relative = relative

    def forward(self, input, target):
        """Forward method for loss function.

        :param torch.Tensor input: Input tensor from real data.
        :param torch.Tensor target: Model tensor output.
        :return: Loss evaluation.
        :rtype: torch.Tensor
        """
        loss = torch.linalg.norm((input-target), ord=self.p, dim=-1)
        if self.relative:
            loss = loss / torch.linalg.norm(input, ord=self.p, dim=-1)          
        return self._reduction(loss)
    


class PowerLoss(LossInterface):
    r"""
    The PowerLoss loss implementation class. Creates a criterion that measures
    the error between each element in the input :math:`x` and
    target :math:`y` powered to a specific integer.

    The unreduced (i.e. with :attr:`reduction` set to ``none``) loss can
    be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{1}{D}\left[\sum_{i=1}^{D} \left| x_n^i - y_n^i \right|^p \right],
    
    If ``'relative'`` is set to true:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{ \sum_{i=1}^{D} | x_n^i - y_n^i|^p }{\sum_{i=1}^{D}|y_n^i|^p},

    where :math:`N` is the batch size. If :attr:`reduction` is not ``none``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets :attr:`reduction` to ``sum``.
    """

    def __init__(self, p=2, reduction = 'mean', relative = False):
        """
        :param int p: Degree of Lp norm. It specifies the type of norm to
            be calculated. See :meth:`torch.linalg.norm` ```'ord'``` to
            see the possible degrees. Default 2 (euclidean norm).
        :param str reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction 
            will be applied, ``'mean'``: the sum of the output will be divided
            by the number of elements in the output, ``'sum'``: the output will
            be summed. Note: :attr:`size_average` and :attr:`reduce` are in the
            process of being deprecated, and in the meantime, specifying either of
            those two args will override :attr:`reduction`. Default: ``'mean'``.
        :param bool relative: Specifies if relative error should be computed.
        """
        super().__init__(reduction=reduction)

        # check consistency
        check_consistency(p, (str,int,float))
        self.p = p
        check_consistency(relative, bool)
        self.relative = relative

    def forward(self, input, target):
        """Forward method for loss function.

        :param torch.Tensor input: Input tensor from real data.
        :param torch.Tensor target: Model tensor output.
        :return: Loss evaluation.
        :rtype: torch.Tensor
        """
        loss = torch.linalg.norm((input-target), ord=self.p, dim=-1)
        if self.relative:
            loss = loss / torch.linalg.norm(input, ord=self.p, dim=-1)          
        return self._reduction(loss)