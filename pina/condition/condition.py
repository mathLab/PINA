""" Condition module. """

from .domain_equation_condition import DomainEquationCondition
from .input_equation_condition import InputPointsEquationCondition
from .input_output_condition import InputOutputPointsCondition
from .data_condition import DataConditionInterface


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
            InputOutputPointsCondition.__slots__ +
            InputPointsEquationCondition.__slots__ +
            DomainEquationCondition.__slots__ +
            DataConditionInterface.__slots__
        )
    )

    def __new__(cls, *args, **kwargs):

        if len(args) != 0:
            raise ValueError(
                "Condition takes only the following keyword "
                f"arguments: {Condition.__slots__}."
            )

        sorted_keys = sorted(kwargs.keys())
        if sorted_keys == sorted(InputOutputPointsCondition.__slots__):
            return InputOutputPointsCondition(**kwargs)
        elif sorted_keys == sorted(InputPointsEquationCondition.__slots__):
            return InputPointsEquationCondition(**kwargs)
        elif sorted_keys == sorted(DomainEquationCondition.__slots__):
            return DomainEquationCondition(**kwargs)
        elif sorted_keys == sorted(DataConditionInterface.__slots__):
            return DataConditionInterface(**kwargs)
        elif sorted_keys == DataConditionInterface.__slots__[0]:
            return DataConditionInterface(**kwargs)
        else:
            raise ValueError(f"Invalid keyword arguments {kwargs.keys()}.")
