from pina.problem import AbstractProblem
from pina import Condition
from pina import Graph


class SupervisedProblem(AbstractProblem):
    """
    A problem definition for supervised learning in PINA.

    This class allows an easy and straightforward definition of a Supervised problem,
    based on a single condition of type `InputTargetCondition`

    :Example:
        >>> import torch
        >>> input_data = torch.rand((100, 10))
        >>> output_data = torch.rand((100, 10))
        >>> problem = SupervisedProblem(input_data, output_data)
    """

    conditions = dict()
    output_variables = None

    def __init__(self, input_, output_):
        """
        Initialize the SupervisedProblem class

        :param input_: Input data of the problem
        :type input_: torch.Tensor | Graph
        :param output_: Output data of the problem
        :type output_: torch.Tensor
        """
        if isinstance(input_, Graph):
            input_ = input_.data
        self.conditions["data"] = Condition(
            input=input_, target=output_
        )
        super().__init__()
