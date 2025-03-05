"""
InputTargetCondition class definition.
"""

import torch
from torch_geometric.data import Data
from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph


class InputTargetCondition(ConditionInterface):
    """
    Condition for domain/equation data. This condition must be used every
    time a Physics Informed or a Supervised Loss is needed in the Solver.
    """

    __slots__ = ["input", "target"]

    def __new__(cls, input, target):
        """
        Instanciate the correct subclass of InputTargetCondition by checking the
        type of the input and target data.

        :param input: torch.Tensor or Graph/Data object containing the input
        :type input: torch.Tensor or Graph or Data
        :param target: torch.Tensor or Graph/Data object containing the target
        :type target: torch.Tensor or Graph or Data
        :return: InputTargetCondition subclass
        :rtype: TensorInputTensorTargetCondition or
            TensorInputGraphTargetCondition or GraphInputTensorTargetCondition
            or GraphInputGraphTargetCondition
        """
        subclass = cls._get_subclass(input, target)
        if subclass is not cls:
            return object.__new__(subclass)
        return super().__new__(cls)

    def __init__(self, input, target):
        """
        Initialize the InputTargetCondition, storing the input and target data.

        :param input: torch.Tensor or Graph/Data object containing the input
        :type input: torch.Tensor or Graph or Data
        :param target: torch.Tensor or Graph/Data object containing the target
        :type target: torch.Tensor or Graph or Data
        """
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
        if is_tensor_input and is_graph_target:
            return TensorInputGraphTargetCondition
        if is_graph_input and is_tensor_target:
            return GraphInputTensorTargetCondition
        if is_graph_input and is_graph_target:
            return GraphInputGraphTargetCondition
        raise ValueError(
            "Invalid input and target types. "
            "Please provide either torch.Tensor or Graph objects."
        )


class TensorInputTensorTargetCondition(InputTargetCondition):
    """
    InputTargetCondition subclass for torch.Tensor input and target data.
    """


class TensorInputGraphTargetCondition(InputTargetCondition):
    """
    InputTargetCondition subclass for torch.Tensor input and Graph/Data target
    data.
    """


class GraphInputTensorTargetCondition(InputTargetCondition):
    """
    InputTargetCondition subclass for Graph/Data input and torch.Tensor target
    data.
    """


class GraphInputGraphTargetCondition(InputTargetCondition):
    """
    InputTargetCondition subclass for Graph/Data input and target data.
    """
