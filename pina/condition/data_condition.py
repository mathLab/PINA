"""
DataCondition class
"""

import torch
from torch_geometric.data import Data
from . import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency


class DataCondition(ConditionInterface):
    """
    Condition for data. This condition must be used every
    time a Unsupervised Loss is needed in the Solver. The conditionalvariable
    can be passed as extra-input when the model learns a conditional
    distribution
    """

    __slots__ = ["input", "conditional_variables"]

    def __new__(cls, input, conditional_variables=None):
        """
        Instanciate the correct subclass of DataCondition by checking the type
        of the input data (input and conditional_variables).

        :param input: torch.Tensor or Graph/Data object containing the input
            data
        :type input: torch.Tensor or Graph or Data
        :param conditional_variables: torch.Tensor or LabelTensor containing
            the conditional variables
        :type conditional_variables: torch.Tensor or LabelTensor
        :return: DataCondition subclass
        :rtype: TensorDataCondition or GraphDataCondition
        """

        if cls == DataCondition:
            subclass = cls._get_subclass(input, conditional_variables)
            return subclass.__new__(subclass, input, conditional_variables)
        return super().__new__(cls)

    def __init__(self, input, conditional_variables=None):
        """
        Initialize the DataCondition, storing the input and conditional
        variables (if any).

        :param input: torch.Tensor or Graph/Data object containing the input
            data
        :type input: torch.Tensor or Graph or Data
        :param conditional_variables: torch.Tensor or LabelTensor containing
            the conditional variables
        :type conditional_variables: torch.Tensor or LabelTensor
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
        if is_graph_input:
            return GraphDataCondition

        raise ValueError(
            "Invalid input types. "
            "Please provide either torch.Tensor or Graph objects."
        )


class TensorDataCondition(DataCondition):
    """
    DataCondition for torch.Tensor input data
    """


class GraphDataCondition(DataCondition):
    """
    DataCondition for Graph/Data input data
    """
