"""Conditions for defining physics and data constraints.

This module provides the interface and implementations for binding mathematical
equations, experimental data, and neural network targets to specific spatial
domains or graph structures. It supports various input-target mappings including
tensor-based, graph-based, and equation-based constraints.
"""

__all__ = [
    "Condition",
    "ConditionInterface",
    "ConditionBase",
    "DomainEquationCondition",
    "InputTargetCondition",
    "InputEquationCondition",
    "DataCondition",
]

from pina._src.condition.condition_interface import ConditionInterface
from pina._src.condition.condition_base import ConditionBase
from pina._src.condition.condition import Condition
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.data_condition import DataCondition
