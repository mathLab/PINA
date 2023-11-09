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
            raise ValueError('equation must be a callable function.'
                             'Expected a callable function, got '
                             f'{equation}')
        self.__equation = equation

    def residual(self, input_, output_, params_=None):
        """
        Residual computation of the equation.

        :param LabelTensor input_: Input points to evaluate the equation.
        :param LabelTensor output_: Output vectors given by my model (e.g,
            a ``FeedForward`` model).
        :return: The residual evaluation of the specified equation.
        :rtype: LabelTensor
        """
        if params_ is not None:
            return self.__equation(input_, output_, params_)
        elif params_ is None:
            return self.__equation(input_, output_)
