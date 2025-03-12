"""
DataCondition class
"""

import torch
from torch_geometric.data import Data
from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph


class DataCondition(ConditionInterface):
    """
    This condition must be used every time a Unsupervised Loss is needed in
    the Solver. The conditionalvariable can be passed as extra-input when
    the model learns a conditional distribution.
    """

    __slots__ = ["input", "conditional_variables"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)
    _avail_conditional_variables_cls = (torch.Tensor, LabelTensor)

    def __new__(cls, input, conditional_variables=None):
        """
        Instantiate the appropriate subclass of DataCondition based on the
        types of input data.

        :param input: Input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph |
            torch_geometric.data.Data | list[Graph] |
            list[torch_geometric.data.Data] | tuple[Graph] |
            tuple[torch_geometric.data.Data]
        :param conditional_variables: Conditional variables for the condition.
        :type conditional_variables: torch.Tensor | LabelTensor
        :return: Subclass of DataCondition.
        :rtype: TensorDataCondition | GraphDataCondition

        :raises ValueError: If input is not of type :class:`torch.Tensor`,
            :class:`LabelTensor`, :class:`Graph`, or
            :class:`torch_geometric.data.Data`.


        """
        if cls != DataCondition:
            return super().__new__(cls)
        if isinstance(input, (torch.Tensor, LabelTensor)):
            subclass = TensorDataCondition
            return subclass.__new__(subclass, input, conditional_variables)

        if isinstance(input, (Graph, Data, list, tuple)):
            cls._check_graph_list_consistency(input)
            subclass = GraphDataCondition
            return subclass.__new__(subclass, input, conditional_variables)

        raise ValueError(
            "Invalid input types. "
            "Please provide either torch_geometric.data.Data or Graph objects."
        )

    def __init__(self, input, conditional_variables=None):
        """
        Initialize the DataCondition, storing the input and conditional
        variables (if any).

        :param input: Input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph |
            torch_geometric.data.Data | list[Graph] |
            list[torch_geometric.data.Data] | tuple[Graph] |
            tuple[torch_geometric.data.Data]
        :param conditional_variables: Conditional variables for the condition.
        :type conditional_variables: torch.Tensor or LabelTensor

        .. note::
            If either `input` is composed by a list of :class:`Graph`/
            :class:`torch_geometric.data.Data` objects, all elements must have
            the same structure (keys and data types)
        """
        super().__init__()
        self.input = input
        self.conditional_variables = conditional_variables


class TensorDataCondition(DataCondition):
    """
    DataCondition for torch.Tensor input data
    """


class GraphDataCondition(DataCondition):
    """
    DataCondition for Graph/Data input data
    """
