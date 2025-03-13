"""Formulation of a Supervised Problem in PINA."""

from ..abstract_problem import AbstractProblem
from ... import Condition
from ... import LabelTensor


class SupervisedProblem(AbstractProblem):
    """
    Definition of a supervised learning problem in PINA.

    This class provides a simple way to define a supervised problem
    using a single condition of type
    :class:`~pina.condition.input_target_condition.InputTargetCondition`.

    :Example:
        >>> import torch
        >>> input_data = torch.rand((100, 10))
        >>> output_data = torch.rand((100, 10))
        >>> problem = SupervisedProblem(input_data, output_data)
    """

    conditions = {}
    output_variables = None
    input_variables = None

    def __init__(
        self, input_, output_, input_variables=None, output_variables=None
    ):
        """
        Initialize the SupervisedProblem class.

        :param input_: Input data of the problem.
        :type input_: torch.Tensor | LabelTensor | Graph | Data
        :param output_: Output data of the problem.
        :type output_: torch.Tensor | LabelTensor | Graph | Data
        """
        # Set input and output variables
        self.input_variables = input_variables
        self.output_variables = output_variables
        # Set the condition
        self.conditions["data"] = Condition(input=input_, target=output_)
        super().__init__()
