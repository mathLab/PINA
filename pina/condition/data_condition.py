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

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        check_consistency(value, (LabelTensor, Graph, torch.Tensor))
        self._data = value

    @property
    def conditionalvariable(self):
        return self._conditionalvariable
    
    @data.setter
    def conditionalvariable(self, value):
        if value is not None:
            check_consistency(value, (LabelTensor, Graph, torch.Tensor))
        self._data = value