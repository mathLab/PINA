
import torch

from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency

class InputOutputPointsCondition(ConditionInterface):
    """
    Condition for domain/equation data. This condition must be used every
    time a Physics Informed or a Supervised Loss is needed in the Solver.
    """

    __slots__ = ["input_points", "output_points"]

    def __init__(self, input_points, output_points):
        """
        TODO
        """
        super().__init__()
        self.input_points = input_points
        self.output_points = output_points
        self.condition_type = ['supervised', 'physics']

    @property
    def input_points(self):
        return self._input_points
    
    @input_points.setter
    def input_points(self, value):
        check_consistency(value, (LabelTensor, Graph, torch.Tensor))
        self._input_points = value

    @property
    def output_points(self):
        return self._output_points
    
    @output_points.setter
    def output_points(self, value):
        check_consistency(value, (LabelTensor, Graph, torch.Tensor))
        self._output_points = value