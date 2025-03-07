"""
This module contains condition classes for supervised learning tasks.
"""

import torch
from torch_geometric.data import Data
from ..label_tensor import LabelTensor
from ..graph import Graph
from .condition_interface import ConditionInterface


class InputTargetCondition(ConditionInterface):
    """
    Condition for domain/equation data. This condition must be used every
    time a Physics Informed or a Supervised Loss is needed in the Solver.
    """

    __slots__ = ["input", "target"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)
    _avail_output_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)

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
        if cls != InputTargetCondition:
            return super().__new__(cls)

        if isinstance(input, (torch.Tensor, LabelTensor)) and isinstance(
            target, (torch.Tensor, LabelTensor)
        ):
            subclass = TensorInputTensorTargetCondition
            return subclass.__new__(subclass, input, target)
        if isinstance(input, (torch.Tensor, LabelTensor)) and isinstance(
            target, (Graph, Data, list, tuple)
        ):
            cls._check_graph_list_consistency(target)
            subclass = TensorInputGraphTargetCondition
            return subclass.__new__(subclass, input, target)

        if isinstance(input, (Graph, Data, list, tuple)) and isinstance(
            target, (torch.Tensor, LabelTensor)
        ):
            cls._check_graph_list_consistency(input)
            subclass = GraphInputTensorTargetCondition
            return subclass.__new__(subclass, input, target)

        if isinstance(input, (Graph, Data, list, tuple)) and isinstance(
            target, (Graph, Data, list, tuple)
        ):
            cls._check_graph_list_consistency(input)
            cls._check_graph_list_consistency(target)
            subclass = GraphInputGraphTargetCondition
            return subclass.__new__(subclass, input, target)

        raise ValueError(
            "Invalid input/target types. "
            "Please provide either Data, Graph, LabelTensor or torch.Tensor "
            "objects."
        )

    def __init__(self, input, target):
        """
        Initialize the InputTargetCondition, storing the input and target data.

        :param input: torch.Tensor or Graph/Data object containing the input
        :type input: torch.Tensor or Graph or Data
        :param target: torch.Tensor or Graph/Data object containing the target
        :type target: torch.Tensor or Graph or Data
        """
        super().__init__()
        self._check_input_target_len(input, target)
        self.input = input
        self.target = target

    @staticmethod
    def _check_input_target_len(input, target):
        if isinstance(input, (Graph, Data)) or isinstance(
            target, (Graph, Data)
        ):
            return
        if len(input) != len(target):
            raise ValueError(
                "The input and target lists must have the same length."
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
