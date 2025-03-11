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
        Instantiate the appropriate subclass of InputEquationCondition based on
        the type of input data.

        :param input: Input data. It can be a LabelTensor or a Graph object.
        :type input: LabelTensor | Graph
        :param EquationInterface equation: Equation object containing the
            equation function.
        :return: Subclass of InputEquationCondition, based on the input type.
        :rtype: InputTensorEquationCondition | InputGraphEquationCondition

        :raises ValueError: If input is not of type :class:`torch.Tensor`,
            :class:`LabelTensor`, :class:`Graph`, or
            :class:`torch_geometric.data.Data`.
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

        :param input: Input data for the condition.
        :type input: torch.Tensor | Graph
        :param EquationInterface equation: Equation object containing the
            equation function.

        .. note::
            If ``input`` is composed by a list of :class:`Graph`/
            :class:`torch_geometric.data.Data` objects, all elements must have
            the same structure (keys and data types). Moreover, at least one
            attribute must be a :class:`LabelTensor`.
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
        Check if at least one LabelTensor is present in the Graph object.

        :param input: Input data.
        :type input: torch.Tensor | Graph | torch_geometric.data.Data

        :raises ValueError: If the input data object does not contain at least
            one LabelTensor.
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
