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
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)
    _avail_conditional_variables_cls = (torch.Tensor, LabelTensor)

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
        if cls == DataCondition and isinstance(
            input, (torch.Tensor, LabelTensor)
        ):
            subclass = TensorDataCondition
            return subclass.__new__(subclass, input, conditional_variables)

        elif cls == DataCondition and isinstance(
            input, (Graph, Data, list, tuple)
        ):
            cls._check_graph_list_consistency(input)
            subclass = GraphDataCondition
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

    def __setattr__(self, key, value):
        if key == "input":
            check_consistency(value, self._avail_input_cls)
            DataCondition.__dict__[key].__set__(self, value)
        elif key == "conditional_variables":
            if value is not None:
                check_consistency(value, self._avail_conditional_variables_cls)
            DataCondition.__dict__[key].__set__(self, value)
        elif key in ("_problem"):
            super().__setattr__(key, value)


class TensorDataCondition(DataCondition):
    """
    DataCondition for torch.Tensor input data
    """


class GraphDataCondition(DataCondition):
    """
    DataCondition for Graph/Data input data
    """
