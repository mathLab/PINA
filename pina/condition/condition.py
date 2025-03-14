"""Module for the Condition class."""

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
    Represents constraints (such as physical equations, boundary conditions,
    etc.) that must be satisfied in a given problem. Condition objects are used
    to formulate the PINA
    :class:`~pina.problem.abstract_problem.AbstractProblem` object.

    There are different types of conditions:

    - :class:`~pina.condition.input_target_condition.InputTargetCondition`:
      Defined by specifying both the input and the target of the condition. In
      this case, the model is trained to produce the target given the input. The
      input and output   data must be one of the :class:`torch.Tensor`,
      :class:`~pina.label_tensor.LabelTensor`,
      :class:`~torch_geometric.data.Data`, or :class:`~pina.graph.Graph`.
      Different implementations exist depending on the type of input and target.
      For more details, see
      :class:`~pina.condition.input_target_condition.InputTargetCondition`.

    - :class:`~pina.condition.domain_equation_condition.DomainEquationCondition`
      : Defined by specifying both the domain and the equation of the condition.
      Here, the model is trained to minimize the equation residual by evaluating
      it at sampled points within the domain.

    - :class:`~pina.condition.input_equation_condition.InputEquationCondition`:
      Defined by specifying the input and the equation of the condition. In this
      case, the model is trained to minimize the equation residual by evaluating
      it at the provided input. The input must be either a
      :class:`~pina.label_tensor.LabelTensor` or a :class:`~pina.graph.Graph`.
      Different implementations exist depending on the type of input. For more
      details, see
      :class:`~pina.condition.input_equation_condition.InputEquationCondition`.

    - :class:`~pina.condition.data_condition.DataCondition`:
      Defined by specifying only the input. In this case, the model is trained
      with an unsupervised custom loss while using the provided data during
      training. The input data must be one of :class:`torch.Tensor`,
      :class:`~pina.label_tensor.LabelTensor`,
      :class:`~torch_geometric.data.Data`, or :class:`~pina.graph.Graph`.
      Additionally, conditional variables can be provided when the model
      depends on extra parameters. These conditional variables must be either
      :class:`torch.Tensor` or :class:`~pina.label_tensor.LabelTensor`.
      Different implementations exist depending on the type of input.
      For more details, see
      :class:`~pina.condition.data_condition.DataCondition`.

    :Example:

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
        Instantiate the appropriate Condition object based on the keyword
        arguments passed.

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
