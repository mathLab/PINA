"""Module for defining the fixed flux equation."""

from pina._src.equation.equation import Equation
from pina._src.core.operator import div


class FixedFlux(Equation):
    """
    Equation to enforce a fixed flux, or divergence, for a specific condition.
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedFlux` class.

        :param float value: The fixed value to be enforced to the flux.
        :param list[str] components: The name of the output variables for which
            the fixed flux condition is applied. It should be a subset of the
            output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        :param list[str] d: The name of the input variables on which the flux
            is computed. It should be a subset of the input labels. If ``None``,
            all the input variables are considered. Default is ``None``.
        """

        def equation(input_, output_):
            """
            Definition of the equation to enforce a fixed flux.

            :param LabelTensor input_: The input points where the residual is
                computed.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :return: The residual values of the equation.
            :rtype: LabelTensor
            """
            return div(output_, input_, components=components, d=d) - value

        super().__init__(equation)
