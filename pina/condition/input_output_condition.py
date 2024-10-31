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
        self._condition_type = ['supervised', 'physics']

    def __setattr__(self, key, value):
        if (key == 'input_points') or (key == 'output_points'):
            check_consistency(value, (LabelTensor, Graph, torch.Tensor))
            InputOutputPointsCondition.__dict__[key].__set__(self, value)
        elif key in ('_problem', '_condition_type'):
            super().__setattr__(key, value)
