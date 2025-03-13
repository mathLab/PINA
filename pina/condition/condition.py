"""Condition module."""

import warnings
from .data_condition import DataCondition
from .domain_equation_condition import DomainEquationCondition
from .input_equation_condition import InputEquationCondition
from .input_target_condition import InputTargetCondition
from ..utils import custom_warning_format

# Set the custom format for warnings
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=DeprecationWarning)


def warning_function(new, old):
    """Handle the deprecation warning.

    :param new: Object to use instead of the old one.
    :type new: str
    :param old: Object to deprecate.
    :type old: str
    """
    warnings.warn(
        f"'{old}' is deprecated and will be removed "
        f"in future versions. Please use '{new}' instead.",
        DeprecationWarning,
    )


class Condition:
    """
    The class ``Condition`` is used to represent the constraints (physical
    equations, boundary conditions, etc.) that should be satisfied in the
    problem at hand. Condition objects are used to formulate the
    PINA :obj:`pina.problem.abstract_problem.AbstractProblem` object.
    Conditions can be specified in four ways:

        1. By specifying the input and target of the condition; in such a
        case, the model is trained to produce the output points given the input
        points. Those points can either be torch.Tensor, LabelTensors, Graph.
        Based on the type of the input and target, there are different
        implementations of the condition. For more details, see
        :class:`~pina.condition.input_target_condition.InputTargetCondition`.

        2. By specifying the domain and the equation of the condition; in such
        a case, the model is trained to minimize the equation residual by
        evaluating it at some samples of the domain.

        3. By specifying the input and the equation of the condition; in
        such a case, the model is trained to minimize the equation residual by
        evaluating it at the passed input points. The input points must be
        a LabelTensor. Based on the type of the input, there are different
        implementations of the condition. For more details, see
        :class:`~pina.condition.input_equation_condition.InputEquationCondition`
        .

        4. By specifying only the input data; in such a case the model is
        trained with an unsupervised costum loss and uses the data in training.
        Additionaly conditioning variables can be passed, whenever the model
        has extra conditioning variable it depends on. Based on the type of the
        input, there are different implementations of the condition. For more
        details, see :class:`~pina.condition.data_condition.DataCondition`.

    Example::

    >>> from pina import Condition
    >>> condition = Condition(
    ...     input=input,
    ...     target=target
    ... )
    >>> condition = Condition(
    ...     domain=location,
    ...     equation=equation
    ... )
    >>> condition = Condition(
    ...     input=input,
    ...     equation=equation
    ... )
    >>> condition = Condition(
    ...     input=data,
    ...     conditional_variables=conditional_variables
    ... )

    """

    __slots__ = list(
        set(
            InputTargetCondition.__slots__
            + InputEquationCondition.__slots__
            + DomainEquationCondition.__slots__
            + DataCondition.__slots__
        )
    )

    def __new__(cls, *args, **kwargs):
        """
        Check the input arguments and return the appropriate Condition object.

        :raises ValueError: If no keyword arguments are passed.
        :raises ValueError: If the keyword arguments are invalid.
        :return: The appropriate Condition object.
        :rtype: ConditionInterface
        """

        if len(args) != 0:
            raise ValueError(
                "Condition takes only the following keyword "
                f"arguments: {Condition.__slots__}."
            )

        # back-compatibility 0.1
        keys = list(kwargs.keys())
        if "location" in keys:
            kwargs["domain"] = kwargs.pop("location")
            warning_function(new="domain", old="location")

        if "input_points" in keys:
            kwargs["input"] = kwargs.pop("input_points")
            warning_function(new="input", old="input_points")

        if "output_points" in keys:
            kwargs["target"] = kwargs.pop("output_points")
            warning_function(new="target", old="output_points")

        sorted_keys = sorted(kwargs.keys())
        if sorted_keys == sorted(InputTargetCondition.__slots__):
            return InputTargetCondition(**kwargs)
        if sorted_keys == sorted(InputEquationCondition.__slots__):
            return InputEquationCondition(**kwargs)
        if sorted_keys == sorted(DomainEquationCondition.__slots__):
            return DomainEquationCondition(**kwargs)
        if (
            sorted_keys == sorted(DataCondition.__slots__)
            or sorted_keys[0] == DataCondition.__slots__[0]
        ):
            return DataCondition(**kwargs)

        raise ValueError(f"Invalid keyword arguments {kwargs.keys()}.")
