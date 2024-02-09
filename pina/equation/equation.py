""" Module for Equation. """

from .equation_interface import EquationInterface


class Equation(EquationInterface):

    def __init__(self, equation):
        """
        Equation class for specifing any equation in PINA.
        Each ``equation`` passed to a ``Condition`` object
        must be an ``Equation`` or ``SystemEquation``.

        :param equation: A ``torch`` callable equation to
            evaluate the residual.
        :type equation: Callable
        """
        if not callable(equation):
            raise ValueError(
                "equation must be a callable function."
                "Expected a callable function, got "
                f"{equation}"
            )
        self.__equation = equation

    def residual(self, input_, output_, params_=None):
        """
        Residual computation of the equation.

        :param LabelTensor input_: Input points to evaluate the equation.
        :param LabelTensor output_: Output vectors given by a model (e.g,
            a ``FeedForward`` model).
        :param dict params_: Dictionary of parameters related to the inverse
            problem (if any).
            If the equation is not related to an ``InverseProblem``, the
            parameters are initialized to ``None`` and the residual is
            computed as ``equation(input_, output_)``.
            Otherwise, the parameters are automatically initialized in the
            ranges specified by the user.

        :return: The residual evaluation of the specified equation.
        :rtype: LabelTensor
        """
        if params_ is None:
            result = self.__equation(input_, output_)
        else:
            result = self.__equation(input_, output_, params_)
        return result
