"""Module for the Equation Interface"""

from abc import ABCMeta, abstractmethod


class EquationInterface(metaclass=ABCMeta):
    """
    Abstract base class for equations.
    """

    @abstractmethod
    def residual(self, input_, output_, params_):
        """
        Abstract method to compute the residual of an equation.

        :param LabelTensor input_: Input points where the equation is evaluated.
        :param LabelTensor output_: Output tensor, eventually produced by a
            :class:`~torch.nn.Module` instance.
        :param dict params_: Dictionary of unknown parameters, associated with a
            :class:`~pina.problem.InverseProblem` instance.
        :return: The computed residual of the equation.
        :rtype: LabelTensor
        """
