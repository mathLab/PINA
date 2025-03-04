import torch

from . import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency
from torch_geometric.data import Data


class DataCondition(ConditionInterface):
    """
    Condition for data. This condition must be used every
    time a Unsupervised Loss is needed in the Solver. The conditionalvariable
    can be passed as extra-input when the model learns a conditional
    distribution
    """

    __slots__ = ["input", "conditional_variables"]

    def __new__(cls, input, conditional_variables=None):
        subclass = cls._get_subclass(input, conditional_variables)
        if subclass is not cls:
            return object.__new__(subclass)
        return super().__new__(cls)

    def __init__(self, input, conditional_variables=None):
        """
        TODO : add docstring
        """
        super().__init__()
        self.input = input
        self.conditional_variables = conditional_variables

    @staticmethod
    def _get_subclass(input, conditional_variables):
        if conditional_variables is not None:
            check_consistency(
                conditional_variables, (torch.Tensor, LabelTensor)
            )
        is_tensor_input = isinstance(input, (LabelTensor, torch.Tensor))
        is_graph_input = isinstance(input, (Data, Graph)) or (
            isinstance(input, list)
            and all(isinstance(i, (Graph, Data)) for i in input)
        )
        if is_tensor_input:
            return TensorDataCondition
        elif is_graph_input:
            return GraphDataCondition
        else:
            raise ValueError(
                "Invalid input types. "
                "Please provide either torch.Tensor or Graph objects."
            )


class TensorDataCondition(DataCondition):
    pass


class GraphDataCondition(DataCondition):
    pass
