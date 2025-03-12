"""
DomainEquationCondition class definition.
"""

from .condition_interface import ConditionInterface
from ..utils import check_consistency
from ..domain import DomainInterface
from ..equation.equation_interface import EquationInterface


class DomainEquationCondition(ConditionInterface):
    """
    Condition defined by a domain and an equation. It can be used in Physics
    Informed problems. Before using this condition, make sure that input data
    are correctly sampled from the domain.
    """

    __slots__ = ["domain", "equation"]

    def __init__(self, domain, equation):
        """
        Initialise the object by storing the domain and equation.

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
