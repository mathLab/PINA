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

    __slots__ = ["data", "conditionalvariable"]

    def __init__(self, data, conditionalvariable=None):
        """
        TODO
        """
        super().__init__()
        self.data = data
        self.conditionalvariable = conditionalvariable
        self.condition_type = 'unsupervised'

    def __setattr__(self, key, value):
        if (key == 'data') or (key == 'conditionalvariable'):
            check_consistency(value, (LabelTensor, Graph, torch.Tensor))
            DataConditionInterface.__dict__[key].__set__(self, value)