"""Module for defining the fixed value equation."""

from pina._src.equation.equation import Equation


class FixedValue(Equation):
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
