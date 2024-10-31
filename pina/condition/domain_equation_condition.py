import torch

from .condition_interface import ConditionInterface
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
        self._condition_type = 'physics'

    def __setattr__(self, key, value):
        if key == 'domain':
            check_consistency(value, (DomainInterface))
            DomainEquationCondition.__dict__[key].__set__(self, value)
        elif key == 'equation':
            check_consistency(value, (EquationInterface))
            DomainEquationCondition.__dict__[key].__set__(self, value)
        elif key in ('_problem', '_condition_type'):
            super().__setattr__(key, value)
