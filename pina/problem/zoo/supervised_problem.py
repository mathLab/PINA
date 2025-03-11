"""Formulation of a Supervised Problem in PINA."""

from ..abstract_problem import AbstractProblem
from ... import Condition
from ... import Graph


class SupervisedProblem(AbstractProblem):
    """
    Definition of a supervised learning problem in PINA.

    This class provides a simple way to define a supervised problem
    using a single condition of type `InputTargetCondition`.

    :Example:
        >>> import torch
        >>> input_data = torch.rand((100, 10))
        >>> output_data = torch.rand((100, 10))
        >>> problem = SupervisedProblem(input_data, output_data)
    """

    conditions = {}
    output_variables = None

    def __init__(self, input_, output_):
        """
        Initialize the SupervisedProblem class.

        :param input_: Input data of the problem.
        :type input_: torch.Tensor | Graph
        :param torch.Tensor output_: Output data of the problem.
        """
        if isinstance(input_, Graph):
            input_ = input_.data
        self.conditions["data"] = Condition(input=input_, target=output_)
        super().__init__()
