"""
Module to define InputEquationCondition class and its subclasses.
"""

import torch
from torch_geometric.data import Data
from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency
from ..equation.equation_interface import EquationInterface


class InputEquationCondition(ConditionInterface):
    """
    Condition for input/equation data. This condition must be used every
    time a Physics Informed Loss is needed in the Solver.
    """

    __slots__ = ["input", "equation"]

    def __new__(cls, input, equation):
        """
        Instanciate the correct subclass of InputEquationCondition by checking
        the type of the input data (only `input`).

        :param input: torch.Tensor or Graph/Data object containing the input
        :type input: torch.Tensor or Graph or Data
        :param EquationInterface equation: Equation object containing the
            equation function
        :return: InputEquationCondition subclass
        :rtype: InputTensorEquationCondition or InputGraphEquationCondition
        """
        check_consistency(equation, (EquationInterface))

        if cls == InputEquationCondition:
            subclass = cls._get_subclass(input)
            return subclass.__new__(subclass, input, equation)
        return super().__new__(cls)

    def __init__(self, input, equation):
        """
        Initialize the InputEquationCondition by storing the input and equation.

        :param input: torch.Tensor or Graph/Data object containing the input
        :type input: torch.Tensor or Graph or Data
        :param EquationInterface equation: Equation object containing the
            equation function
        """
        super().__init__()
        self.input = input
        self.equation = equation

    @staticmethod
    def _get_subclass(input):
        is_tensor_input = isinstance(input, (LabelTensor, torch.Tensor))
        is_graph_input = isinstance(input, (Data, Graph)) or (
            isinstance(input, list)
            and all(isinstance(i, (Graph, Data)) for i in input)
        )
        if is_tensor_input:
            return InputTensorEquationCondition
        if is_graph_input:
            return InputGraphEquationCondition
        raise ValueError(
            "Invalid input types. "
            "Please provide either torch.Tensor or Graph objects."
        )


class InputTensorEquationCondition(InputEquationCondition):
    """
    InputEquationCondition subclass for torch.Tensor input data.
    """


class InputGraphEquationCondition(InputEquationCondition):
    """
    InputEquationCondition subclass for Graph input data.
    """

    def __init__(self, input, equation):
        super().__init__(input, equation)
        self._check_graph_list_consistency(input)
