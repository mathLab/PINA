"""Module for defining various general equations."""

from pina._src.equation.equation import Equation
from pina._src.core.operator import grad, div, laplacian


class FixedValue(Equation):  # pylint: disable=R0903
    """
    Equation to enforce a fixed value. Can be used to enforce Dirichlet Boundary
    conditions.
    """

    def __init__(self, value, components=None):
        """
        Initialization of the :class:`FixedValue` class.

        :param float value: The fixed value to be enforced.
        :param list[str] components: The name of the output variables for which
            the fixed value condition is applied. It should be a subset of the
            output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        """

        def equation(_, output_):
            """
            Definition of the equation to enforce a fixed value.

            :param LabelTensor input_: The input points where the residual is
                computed.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :return: The residual values of the equation.
            :rtype: LabelTensor
            """
            if components is None:
                return output_ - value
            return output_.extract(components) - value

        super().__init__(equation)


class FixedGradient(Equation):  # pylint: disable=R0903
    """
    Equation to enforce a fixed gradient for a specific condition.
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedGradient` class.

        :param float value: The fixed value to be enforced to the gradient.
        :param list[str] components: The name of the output variables for which
            the fixed gradient condition is applied. It should be a subset of
            the output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        :param list[str] d: The name of the input variables on which the
            gradient is computed. It should be a subset of the input labels.
            If ``None``, all the input variables are considered.
            Default is ``None``.
        """

        def equation(input_, output_):
            """
            Definition of the equation to enforce a fixed gradient.

            :param LabelTensor input_: The input points where the residual is
                computed.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :return: The residual values of the equation.
            :rtype: LabelTensor
            """
            return grad(output_, input_, components=components, d=d) - value

        super().__init__(equation)


class FixedFlux(Equation):  # pylint: disable=R0903
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


class FixedLaplacian(Equation):  # pylint: disable=R0903
    """
    Equation to enforce a fixed laplacian for a specific condition.
    """

    def __init__(self, value, components=None, d=None):
        """
        Initialization of the :class:`FixedLaplacian` class.

        :param float value: The fixed value to be enforced to the laplacian.
        :param list[str] components: The name of the output variables for which
            the fixed laplace condition is applied. It should be a subset of the
            output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        :param list[str] d: The name of the input variables on which the
            laplacian is computed. It should be a subset of the input labels.
            If ``None``, all the input variables are considered.
            Default is ``None``.
        """

        def equation(input_, output_):
            """
            Definition of the equation to enforce a fixed laplacian.

            :param LabelTensor input_: The input points where the residual is
                computed.
            :param LabelTensor output_: The output tensor, potentially produced
                by a :class:`torch.nn.Module` instance.
            :return: The residual values of the equation.
            :rtype: LabelTensor
            """
            return (
                laplacian(output_, input_, components=components, d=d) - value
            )

        super().__init__(equation)


class Laplace(FixedLaplacian):  # pylint: disable=R0903
    r"""
    Equation to enforce a null laplacian for a specific condition.
    The equation is defined as follows:

    .. math::

        \Delta u = 0

    """

    def __init__(self, components=None, d=None):
        """
        Initialization of the :class:`Laplace` class.

        :param list[str] components: The name of the output variables for which
            the null laplace condition is applied. It should be a subset of the
            output labels. If ``None``, all output variables are considered.
            Default is ``None``.
        :param list[str] d: The name of the input variables on which the
            laplacian is computed. It should be a subset of the input labels.
            If ``None``, all the input variables are considered.
            Default is ``None``.
        """
        super().__init__(0.0, components=components, d=d)
