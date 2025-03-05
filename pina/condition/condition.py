"""Condition module."""

from .domain_equation_condition import DomainEquationCondition
from .input_equation_condition import InputEquationCondition
from .input_target_condition import InputTargetCondition
from .data_condition import DataCondition
import warnings
from ..utils import custom_warning_format

# Set the custom format for warnings
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=DeprecationWarning)


class Condition:
    """
    The class ``Condition`` is used to represent the constraints (physical
    equations, boundary conditions, etc.) that should be satisfied in the
    problem at hand. Condition objects are used to formulate the
    PINA :obj:`pina.problem.abstract_problem.AbstractProblem` object.
    Conditions can be specified in four ways:

        1. By specifying the input and output points of the condition; in such a
        case, the model is trained to produce the output points given the input
        points. Those points can either be torch.Tensor, LabelTensors, Graph

        2. By specifying the location and the equation of the condition; in such
        a case, the model is trained to minimize the equation residual by
        evaluating it at some samples of the location.

        3. By specifying the input points and the equation of the condition; in
        such a case, the model is trained to minimize the equation residual by
        evaluating it at the passed input points. The input points must be
        a LabelTensor.

        4. By specifying only the data matrix; in such a case the model is
        trained with an unsupervised costum loss and uses the data in training.
        Additionaly conditioning variables can be passed, whenever the model
        has extra conditioning variable it depends on.

    Example::

    >>> TODO

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

        if len(args) != 0:
            raise ValueError(
                "Condition takes only the following keyword "
                f"arguments: {Condition.__slots__}."
            )

        # back-compatibility 0.1
        if "location" in kwargs.keys():
            kwargs["domain"] = kwargs.pop("location")
            warnings.warn(
                f"'location' is deprecated and will be removed "
                f"in future versions. Please use 'domain' instead.",
                DeprecationWarning,
            )

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
