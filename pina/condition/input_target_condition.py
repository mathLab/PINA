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
    Condition defined by input and target data. This condition can be used in
    both supervised learning and Physics-informed problems. Based on the type of
    the input and target, different condition implementations are available:

    - :class:`TensorInputTensorTargetCondition`: For :class:`torch.Tensor` or
        :class:`~pina.label_tensor.LabelTensor` input and target data.
    - :class:`TensorInputGraphTargetCondition`: For :class:`torch.Tensor` or
        :class:`~pina.label_tensor.LabelTensor` input and
        :class:`~pina.graph.Graph` or :class:`~torch_geometric.data.Data`
        target data.
    - :class:`GraphInputTensorTargetCondition`: For :class:`~pina.graph.Graph`
        or :class:`~torch_geometric.data.Data` input and :class:`torch.Tensor`
        or :class:`~pina.label_tensor.LabelTensor` target data.
    - :class:`GraphInputGraphTargetCondition`: For :class:`~pina.graph.Graph` or
        :class:`~torch_geometric.data.Data` input and target data.
    """

    __slots__ = ["input", "target"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)
    _avail_output_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)

    def __new__(cls, input, target):
        """
        Instantiate the appropriate subclass of InputTargetCondition based on
        the types of input and target data.

        :param input: Input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param target: Target data for the condition.
        :type target: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :return: Subclass of InputTargetCondition
        :rtype: pina.condition.input_target_condition.
            TensorInputTensorTargetCondition |
            pina.condition.input_target_condition.
            TensorInputGraphTargetCondition |
            pina.condition.input_target_condition.
            GraphInputTensorTargetCondition |
            pina.condition.input_target_condition.GraphInputGraphTargetCondition

        :raises ValueError: If ``input`` and/or ``target`` are not of type
            :class:`torch.Tensor`, :class:`~pina.label_tensor.LabelTensor`,
            :class:`~pina.graph.Graph`, or :class:`~torch_geometric.data.Data`.
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
            "Please provide either torch_geometric.data.Data, Graph, "
            "LabelTensor or torch.Tensor objects."
        )

    def __init__(self, input, target):
        """
        Initialize the object by storing the ``input`` and ``target`` data.

        :param input: Input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param target: Target data for the condition.
        :type target: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]

        .. note::
            If either ``input`` or ``target`` are composed by a list of
            :class:`~pina.graph.Graph` or :class:`~torch_geometric.data.Data`
            objects, all elements must have the same structure (keys and data
            types)
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
    InputTargetCondition subclass for :class:`torch.Tensor` or
    :class:`~pina.label_tensor.LabelTensor` ``input`` and ``target`` data.
    """


class TensorInputGraphTargetCondition(InputTargetCondition):
    """
    InputTargetCondition subclass for :class:`torch.Tensor` or
    :class:`~pina.label_tensor.LabelTensor` ``input`` and
    :class:`~pina.graph.Graph` or :class:`~torch_geometric.data.Data` `target`
    data.
    """


class GraphInputTensorTargetCondition(InputTargetCondition):
    """
    InputTargetCondition subclass for :class:`~pina.graph.Graph` o
    :class:`~torch_geometric.data.Data` ``input`` and :class:`torch.Tensor` or
    :class:`~pina.label_tensor.LabelTensor` ``target`` data.
    """


class GraphInputGraphTargetCondition(InputTargetCondition):
    """
    InputTargetCondition subclass for :class:`~pina.graph.Graph`/
    :class:`~torch_geometric.data.Data` ``input`` and ``target`` data.
    """
