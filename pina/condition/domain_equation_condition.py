import torch

from .condition_interface import ConditionInterface
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..utils import check_consistency
from ..domain import DomainInterface
from ..equation.equation_interface import EquationInterface

class DomainEquationCondition(ConditionInterface):
    """
    Condition for domain/equation data. This condition must be used every
    time a Physics Informed Loss is needed in the Solver.
    """

    __slots__ = ["domain", "equation"]

    def __init__(self, domain, equation):
        """
        TODO
        """
        super().__init__()
        self.domain = domain
        self.equation = equation
        self.condition_type = 'physics'

    @property
    def domain(self):
        return self._domain
    
    @domain.setter
    def domain(self, value):
        check_consistency(value, (DomainInterface))
        self._domain = value

    @property
    def equation(self):
        return self._equation
    
    @equation.setter
    def equation(self, value):
        check_consistency(value, (EquationInterface))
        self._equation = value