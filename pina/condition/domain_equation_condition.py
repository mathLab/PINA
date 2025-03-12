"""
DomainEquationCondition class definition.
"""

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
        Initialize the object by storing the domain and equation.

        :param DomainInterface domain: Domain object containing the domain data.
        :param EquationInterface equation: Equation object containing the
            equation data.
        """
        super().__init__()
        self.domain = domain
        self.equation = equation

    def __setattr__(self, key, value):
        if key == "domain":
            check_consistency(value, (DomainInterface, str))
            DomainEquationCondition.__dict__[key].__set__(self, value)
        elif key == "equation":
            check_consistency(value, (EquationInterface))
            DomainEquationCondition.__dict__[key].__set__(self, value)
        elif key in ("_problem"):
            super().__setattr__(key, value)
