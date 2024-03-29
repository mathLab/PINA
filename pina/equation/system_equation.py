""" Module for SystemEquation. """

import torch
from .equation import Equation
from ..utils import check_consistency


class SystemEquation(Equation):

    def __init__(self, list_equation, reduction="mean"):
        """
        System of Equation class for specifing any system
        of equations in PINA.
        Each ``equation`` passed to a ``Condition`` object
        must be an ``Equation`` or ``SystemEquation``.
        A ``SystemEquation`` is specified by a list of
        equations.

        :param Callable equation: A ``torch`` callable equation to
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
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif (reduction == "none") or callable(reduction):
            self.reduction = reduction
        else:
            raise NotImplementedError(
                "Only mean and sum reductions implemented."
            )

    def residual(self, input_, output_, params_=None):
        """
        Residual computation for the equations of the system.

        :param LabelTensor input_: Input points to evaluate the system of
            equations.
        :param LabelTensor output_: Output vectors given by a model (e.g,
            a ``FeedForward`` model).
        :param dict params_: Dictionary of parameters related to the inverse
            problem (if any).
            If the equation is not related to an ``InverseProblem``, the
            parameters are initialized to ``None`` and the residual is
            computed as ``equation(input_, output_)``.
            Otherwise, the parameters are automatically initialized in the
            ranges specified by the user.

        :return: The residual evaluation of the specified system of equations,
            aggregated by the ``reduction`` defined in the ``__init__``.
        :rtype: LabelTensor
        """
        residual = torch.hstack(
            [
                equation.residual(input_, output_, params_)
                for equation in self.equations
            ]
        )

        if self.reduction == "none":
            return residual

        return self.reduction(residual, dim=-1)
