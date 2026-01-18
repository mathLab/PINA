"""Conditions for defining physics and data constraints.

This module provides the interface and implementations for binding mathematical
equations, experimental data, and neural network targets to specific spatial
domains or graph structures. It supports various input-target mappings including
tensor-based, graph-based, and equation-based constraints.
"""

__all__ = [
    "Condition",
    "ConditionInterface",
    "DomainEquationCondition",
    "InputTargetCondition",
    "TensorInputTensorTargetCondition",
    "TensorInputGraphTargetCondition",
    "GraphInputTensorTargetCondition",
    "GraphInputGraphTargetCondition",
    "InputEquationCondition",
    "InputTensorEquationCondition",
    "InputGraphEquationCondition",
    "DataCondition",
    "GraphDataCondition",
    "TensorDataCondition",
]

from pina._src.condition.condition_interface import ConditionInterface
from pina._src.condition.condition import Condition
from pina._src.condition.domain_equation_condition import DomainEquationCondition
from pina._src.condition.input_target_condition import (
    InputTargetCondition,
    TensorInputTensorTargetCondition,
    TensorInputGraphTargetCondition,
    GraphInputTensorTargetCondition,
    GraphInputGraphTargetCondition,
)
from pina._src.condition.input_equation_condition import (
    InputEquationCondition,
    InputTensorEquationCondition,
    InputGraphEquationCondition,
)
from pina._src.condition.data_condition import (
    DataCondition,
    GraphDataCondition,
    TensorDataCondition,
)
