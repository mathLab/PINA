"""
Module to define InputEquationCondition class and its subclasses.
"""

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
    _avail_input_cls = (LabelTensor, Graph, list, tuple)
    _avail_equation_cls = EquationInterface

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

        # If the class is already a subclass, return the instance
        if cls != InputEquationCondition:
            return super().__new__(cls)

        # Instanciate the correct subclass
        if isinstance(input, (Graph, Data, list, tuple)):
            subclass = InputGraphEquationCondition
            cls._check_graph_list_consistency(input)
            subclass._check_label_tensor(input)
            return subclass.__new__(subclass, input, equation)
        if isinstance(input, LabelTensor):
            subclass = InputTensorEquationCondition
            return subclass.__new__(subclass, input, equation)

        # If the input is not a LabelTensor or a Graph object raise an error
        raise ValueError(
            "The input data object must be a LabelTensor or a Graph object."
        )

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

    def __setattr__(self, key, value):
        if key == "input":
            check_consistency(value, self._avail_input_cls)
            InputEquationCondition.__dict__[key].__set__(self, value)
        elif key == "equation":
            check_consistency(value, self._avail_equation_cls)
            InputEquationCondition.__dict__[key].__set__(self, value)
        elif key in ("_problem"):
            super().__setattr__(key, value)


class InputTensorEquationCondition(InputEquationCondition):
    """
    InputEquationCondition subclass for LabelTensor input data.
    """


class InputGraphEquationCondition(InputEquationCondition):
    """
    InputEquationCondition subclass for Graph input data.
    """

    @staticmethod
    def _check_label_tensor(input):
        """
        Check if the input is a LabelTensor.

        :param input: input data
        :type input: torch.Tensor or Graph or Data
        """
        # Store the fist element of the list/tuple if input is a list/tuple
        # it is anougth to check the first element because all elements must
        # have the same type and structure (already checked)
        data = input[0] if isinstance(input, (list, tuple)) else input

        # Check if the input data contains at least one LabelTensor
        for v in data.values():
            if isinstance(v, LabelTensor):
                return
        raise ValueError(
            "The input data object must contain at least one LabelTensor."
        )
