"""Module for EquationInterface class"""

from abc import ABCMeta, abstractmethod


class EquationInterface(metaclass=ABCMeta):
    """
    The abstract `AbstractProblem` class. All the class defining a PINA Problem
    should be inheritied from this class.

    In the definition of a PINA problem, the fundamental elements are:
    the output variables, the condition(s), and the domain(s) where the
    conditions are applied.
    """

    @abstractmethod
    def residual(self, input_, output_, params_):
        """
        Residual computation of the equation.

        :param LabelTensor input_: Input points to evaluate the equation.
        :param LabelTensor output_: Output vectors given by my model (e.g., a ``FeedForward`` model).
        :param dict params_: Dictionary of unknown parameters, eventually
            related to an ``InverseProblem``.
        :return: The residual evaluation of the specified equation.
        :rtype: LabelTensor
        """
        pass
