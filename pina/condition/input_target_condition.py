import torch

from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from torch_geometric.data import Data


class InputTargetCondition(ConditionInterface):
    """
    Condition for domain/equation data. This condition must be used every
    time a Physics Informed or a Supervised Loss is needed in the Solver.
    """

    __slots__ = ["input", "target"]

    def __new__(cls, input, target):
        subclass = cls._get_subclass(input, target)
        if subclass is not cls:
            return object.__new__(subclass)
        return super().__new__(cls)

    def __init__(self, input, target):
        super().__init__()
        self.input = input
        self.target = target

    @staticmethod
    def _get_subclass(input, target):
        is_tensor_input = isinstance(input, (torch.Tensor, LabelTensor))
        is_tensor_target = isinstance(target, (torch.Tensor, LabelTensor))

        is_graph_input = isinstance(input, (Data, Graph)) or (
            isinstance(input, list)
            and all(isinstance(i, (Graph, Data)) for i in input)
        )
        is_graph_target = isinstance(target, (Data, Graph)) or (
            isinstance(target, list)
            and all(isinstance(i, (Graph, Data)) for i in target)
        )

        if is_tensor_input and is_tensor_target:
            return TensorInputTensorTargetCondition
        elif is_tensor_input and is_graph_target:
            return TensorInputGraphTargetCondition
        elif is_graph_input and is_tensor_target:
            return GraphInputTensorTargetCondition
        elif is_graph_input and is_graph_target:
            return GraphInputGraphTargetCondition
        else:
            raise ValueError(
                "Invalid input and target types. "
                "Please provide either torch.Tensor or Graph objects."
            )

    def __init__(self, input, target):
        """
        TODO : add docstring
        """
        super().__init__()
        self.input = input
        self.target = target


class TensorInputTensorTargetCondition(InputTargetCondition):
    pass


class TensorInputGraphTargetCondition(InputTargetCondition):
    pass


class GraphInputTensorTargetCondition(InputTargetCondition):
    pass


class GraphInputGraphTargetCondition(InputTargetCondition):
    pass
