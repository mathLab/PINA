import torch

from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency
from ..equation.equation_interface import EquationInterface
from torch_geometric.data import Data


class InputEquationCondition(ConditionInterface):
    """
    Condition for input/equation data. This condition must be used every
    time a Physics Informed Loss is needed in the Solver.
    """

    __slots__ = ["input", "equation"]

    def __new__(cls, input, equation):
        subclass = cls._get_subclass(input, equation)
        if subclass is not cls:
            return object.__new__(subclass)
        return super().__new__(cls)

    def __init__(self, input, equation):
        """
        TODO : add docstring
        """
        super().__init__()
        self.input = input
        self.equation = equation

    @staticmethod
    def _get_subclass(input, equation):
        check_consistency(equation, (EquationInterface))
        is_tensor_input = isinstance(input, (LabelTensor, torch.Tensor))
        is_graph_input = isinstance(input, (Data, Graph)) or (
            isinstance(input, list)
            and all(isinstance(i, (Graph, Data)) for i in input)
        )
        if is_tensor_input:
            return InputTensorEquationCondition
        elif is_graph_input:
            return InputGraphEquationCondition
        else:
            raise ValueError(
                "Invalid input types. "
                "Please provide either torch.Tensor or Graph objects."
            )


class InputTensorEquationCondition(InputEquationCondition):
    pass


class InputGraphEquationCondition(InputEquationCondition):
    pass
