"""Module for the System of Equation."""

import torch
from .equation_interface import EquationInterface
from .equation import Equation
from ..utils import check_consistency


class SystemEquation(EquationInterface):
    """
    Implementation of the System of Equations. Every ``equation`` passed to a
    :class:`~pina.condition.Condition` object must be either a :class:`Equation`
    or a :class:`~pina.equation.SystemEquation` instance.
    """

    def __init__(self, list_equation, reduction=None):
        """
        Initialization of the :class:`SystemEquation` class.

        :param Callable equation: A ``torch`` callable function used to compute
            the residual of a mathematical equation.
        :param str reduction: The reduction method to aggregate the residuals of
            each equation. Available options are: ``None``, ``mean``, ``sum``,
            ``callable``.
            If ``None``, no reduction is applied. If ``mean``, the output sum is
            divided by the number of elements in the output. If ``sum``, the
            output is summed. ``callable`` is a user-defined callable function
            to perform reduction, no checks guaranteed. Default is ``None``.
        :raises NotImplementedError: If the reduction is not implemented.
        """
        check_consistency([list_equation], list)

        # equations definition
        self.equations = []
        for _, equation in enumerate(list_equation):
            self.equations.append(Equation(equation))

        # possible reduction
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif (reduction is None) or callable(reduction):
            self.reduction = reduction
        else:
            raise NotImplementedError(
                "Only mean and sum reductions implemented."
            )

    def residual(self, input_, output_, params_=None):
        """
        Compute the residual for each equation in the system of equations and
        aggregate it according to the ``reduction`` specified in the
        ``__init__`` method.

        :param LabelTensor input_: Input points where each equation of the
            system is evaluated.
        :param LabelTensor output_: Output tensor, eventually produced by a
            :class:`~torch.nn.Module` instance.
        :param dict params_: Dictionary of unknown parameters, associated with a
            :class:`~pina.problem.InverseProblem` instance. If the equation is
            not related to a :class:`~pina.problem.InverseProblem` instance, the
            parameters must be initialized to ``None``. Default is ``None``.

        :return: The aggregated residuals of the system of equations.
        :rtype: LabelTensor
        """
        residual = torch.hstack(
            [
                equation.residual(input_, output_, params_)
                for equation in self.equations
            ]
        )

        if self.reduction is None:
            return residual

        return self.reduction(residual, dim=-1)
