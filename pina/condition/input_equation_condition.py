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
        self.condition_type = 'physics'

    @property
    def input_points(self):
        return self._input_points
    
    @input_points.setter
    def input_points(self, value):
        check_consistency(value, (LabelTensor)) # for now only labeltensors, we need labels for the operators!
        self._input_points = value

    @property
    def equation(self):
        return self._equation
    
    @equation.setter
    def equation(self, value):
        check_consistency(value, (EquationInterface))
        self._equation = value