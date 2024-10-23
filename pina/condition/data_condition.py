import torch

from . import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency

class DataConditionInterface(ConditionInterface):
    """
    Condition for data. This condition must be used every
    time a Unsupervised Loss is needed in the Solver. The conditionalvariable
    can be passed as extra-input when the model learns a conditional
    distribution
    """

    __slots__ = ["input_points", "conditional_variables"]

    def __init__(self, input_points, conditional_variables=None):
        """
        TODO
        """
        super().__init__()
        self.input_points = input_points
        self.conditional_variables = conditional_variables
        self._condition_type = 'unsupervised'

    def __setattr__(self, key, value):
        if (key == 'input_points') or (key == 'conditional_variables'):
            check_consistency(value, (LabelTensor, Graph, torch.Tensor))
            DataConditionInterface.__dict__[key].__set__(self, value)
        elif key in ('_problem', '_condition_type'):
            super().__setattr__(key, value)