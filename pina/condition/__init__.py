"""Conditions for defining physics and data constraints.

This module provides the interface and implementations for binding mathematical
equations, experimental data, and neural network targets to specific spatial
domains or graph structures. It supports various input-target mappings including
tensor-based, graph-based, and equation-based constraints.

:Example:

    >>> from pina.condition import InputTargetCondition
    >>> import torch
    >>> condition = InputTargetCondition(
    ...     input_points=torch.rand(10, 2),
    ...     target_points=torch.rand(10, 1),
    ... )
"""

__all__ = [
    "ConditionInterface",
    "BaseCondition",
    "Condition",
    "DomainEquationCondition",
    "InputTargetCondition",
    "InputEquationCondition",
    "DataCondition",
    "TimeSeriesCondition",
    "GraphTimeSeriesCondition",
]

from pina._src.condition.condition_interface import ConditionInterface
from pina._src.condition.base_condition import BaseCondition
from pina._src.condition.condition import Condition
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.data_condition import DataCondition
from pina._src.condition.time_series_condition import TimeSeriesCondition
from pina._src.condition.graph_time_series_condition import (
    GraphTimeSeriesCondition,
)
