""" Module """
import torch
from .equation import Equation
from ..utils import check_consistency


class SystemEquation(Equation):

    def __init__(self, list_equation, reduction='mean'):
        """
        System of Equation class for specifing any system
        of equations in PINA.
        Each ``equation`` passed to a ``Condition`` object
        must be an ``Equation`` or ``SystemEquation``. 
        A ``SystemEquation`` is specified by a list of 
        equations.

        :param callable equation: A ``torch`` callable equation to
            evaluate the residual
        :param str reduction: Specifies the reduction to apply to the output:
            ``none`` | ``mean`` | ``sum``  | ``callable``. ``none``: no reduction 
            will be applied, ``mean``: the sum of the output will be divided
            by the number of elements in the output, ``sum``: the output will
            be summed. ``callable`` a callable function to perform reduction,
            no checks guaranteed. Default: ``mean``.
        """
        check_consistency([list_equation], list)
        check_consistency(reduction, str)

        # equations definition
        self.equations = []
        for _, equation in enumerate(list_equation):
            self.equations.append(Equation(equation))

        # possible reduction
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        elif (reduction == 'none') or callable(reduction):
            self.reduction = reduction
        else:
            raise NotImplementedError(
                'Only mean and sum reductions implemented.')

    def residual(self, input_, output_):
        """
        Residual computation of the equation.

        :param LabelTensor input_: Input points to evaluate the equation.
        :param LabelTensor output_: Output vectors given my a model (e.g,
            a ``FeedForward`` model).
        :return: The residual evaluation of the specified equation,
            aggregated by the ``reduction`` defined in the ``__init__``.
        :rtype: LabelTensor
        """
        residual = torch.hstack(
            [equation.residual(input_, output_) for equation in self.equations])

        if self.reduction == 'none':
            return residual

        return self.reduction(residual, dim=-1)
