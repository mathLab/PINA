"""Module for the Equation Interface."""

from abc import ABCMeta, abstractmethod


class EquationInterface(metaclass=ABCMeta):
    """
    Abstract interface for all equations.

    :Example:

        >>> # This class is not meant to be instantiated directly.
        >>> # Use specific equation implementations instead:
        >>> from pina.equation import PoissonEquation
        >>> eq = PoissonEquation(forcing_term=lambda x: x**2)
    """

    @abstractmethod
    def residual(self, input_, output_, params_=None):
        """
        Evaluate the equation residual at the given inputs.

        :param LabelTensor input_: The input points where the residual is
            computed.
        :param LabelTensor output_: The output tensor, potentially produced by a
            :class:`torch.nn.Module` instance.
        :param dict params_: An optional dictionary of unknown parameters, used
            in :class:`~pina.problem.inverse_problem.InverseProblem` settings.
            If the equation is not related to an inverse problem, this should be
            set to ``None``. Default is ``None``.
        :return: The residual values of the equation.
        :rtype: LabelTensor
        """

    @abstractmethod
    def to(self, device):
        """
        Move all tensor attributes to the specified device.

        :param torch.device device: The target device to move the tensors to.
        :return: The instance moved to the specified device.
        :rtype: EquationInterface
        """
