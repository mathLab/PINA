"""Module for the Equation Interface."""

from abc import ABCMeta, abstractmethod


class EquationInterface(metaclass=ABCMeta):
    """
    Abstract base class for equations.

    Equations in PINA simplify the training process. When defining a problem,
    each equation passed to a :class:`~pina.condition.condition.Condition`
    object must be either an :class:`~pina.equation.equation.Equation` or a
    :class:`~pina.equation.system_equation.SystemEquation` instance.

    An :class:`~pina.equation.equation.Equation` is a wrapper for a callable
    function, while :class:`~pina.equation.system_equation.SystemEquation`
    wraps a list of callable functions. To streamline code writing, PINA
    provides a diverse set of pre-implemented equations, such as
    :class:`~pina.equation.equation_factory.FixedValue`,
    :class:`~pina.equation.equation_factory.FixedGradient`, and many others.
    """

    @abstractmethod
    def residual(self, input_, output_, params_):
        """
        Abstract method to compute the residual of an equation.

        :param LabelTensor input_: Input points where the equation is evaluated.
        :param LabelTensor output_: Output tensor, eventually produced by a
            :class:`torch.nn.Module` instance.
        :param dict params_: Dictionary of unknown parameters, associated with a
            :class:`~pina.problem.inverse_problem.InverseProblem` instance.
        :return: The computed residual of the equation.
        :rtype: LabelTensor
        """
