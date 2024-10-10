import torch

from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency
from ..equation.equation_interface import EquationInterface

class InputPointsEquationCondition(ConditionInterface):
    """
    Condition for input_points/equation data. This condition must be used every
    time a Physics Informed Loss is needed in the Solver.
    """

    __slots__ = ["input_points", "equation"]

    def __init__(self, input_points, equation):
        """
        TODO
        """
        super().__init__()
        self.input_points = input_points
        self.equation = equation
        self._condition_type = 'physics'

    def __setattr__(self, key, value):
        if key == 'input_points':
            check_consistency(value, (LabelTensor)) # for now only labeltensors, we need labels for the operators!
            InputPointsEquationCondition.__dict__[key].__set__(self, value)
        elif key == 'equation':
            check_consistency(value, (EquationInterface))
            InputPointsEquationCondition.__dict__[key].__set__(self, value)
        elif key in ('_condition_type', '_problem', 'problem', 'condition_type'):
            super().__setattr__(key, value)