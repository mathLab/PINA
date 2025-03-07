"""Module for defining different equations."""

from .equation import Equation
from ..operator import grad, div, laplacian


class FixedValue(Equation):
    """
    Fixed Value Equation class.
    """

    def __init__(self, value, components=None):
        """
        This class can be
        used to enforced a fixed value for a specific
        condition, e.g. Dirichlet Boundary conditions.

        :param float value: Value to be mantained fixed.
        :param list(str) components: the name of the output
            variables to calculate the gradient for. It should
            be a subset of the output labels. If ``None``,
            all the output variables are considered.
            Default is ``None``.
        """

        # pylint: disable=W0613
        def equation(input_, output_):
            if components is None:
                return output_ - value
            return output_.extract(components) - value

        super().__init__(equation)


class FixedGradient(Equation):
    """
    Fixed Gradient Equation class.
    """

    def __init__(self, value, components=None, d=None):
        """
        This class can beused to enforced a fixed gradient for a specific
        condition.

        :param float value: Value to be mantained fixed.
        :param list(str) components: the name of the output
            variables to calculate the gradient for. It should
            be a subset of the output labels. If ``None``,
            all the output variables are considered.
            Default is ``None``.
        :param list(str) d: the name of the input variables on
            which the gradient is calculated. d should be a subset
            of the input labels. If ``None``, all the input variables
            are considered. Default is ``None``.
        """

        def equation(input_, output_):
            return grad(output_, input_, components=components, d=d) - value

        super().__init__(equation)


class FixedFlux(Equation):
    """
    Fixed Flux Equation class.
    """

    def __init__(self, value, components=None, d=None):
        """
        This class can be used to enforced a fixed flux for a specific
        condition.

        :param float value: Value to be mantained fixed.
        :param list(str) components: the name of the output
            variables to calculate the flux for. It should
            be a subset of the output labels. If ``None``,
            all the output variables are considered.
            Default is ``None``.
        :param list(str) d: the name of the input variables on
            which the flux is calculated. d should be a subset
            of the input labels. If ``None``, all the input variables
            are considered. Default is ``None``.
        """

        def equation(input_, output_):
            return div(output_, input_, components=components, d=d) - value

        super().__init__(equation)


class Laplace(Equation):
    """
    Laplace Equation class.
    """

    def __init__(self, components=None, d=None):
        """
        This class can be
        used to enforced a Laplace equation for a specific
        condition (force term set to zero).

        :param list(str) components: the name of the output
            variables to calculate the flux for. It should
            be a subset of the output labels. If ``None``,
            all the output variables are considered.
            Default is ``None``.
        :param list(str) d: the name of the input variables on
            which the flux is calculated. d should be a subset
            of the input labels. If ``None``, all the input variables
            are considered. Default is ``None``.
        """

        def equation(input_, output_):
            return laplacian(output_, input_, components=components, d=d)

        super().__init__(equation)
