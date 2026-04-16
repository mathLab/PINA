"""
Mathematical equations and physical laws.

This module provides a framework for defining differential equations,
boundary conditions, and complex systems of equations. It includes
pre-defined physical models such as Poisson, Laplace, and Wave equations,
along with factories for common derivative-based constraints.
"""

__all__ = [
    "EquationInterface",
    "BaseEquation",
    "Equation",
    "SystemEquation",
    "FixedValue",
    "FixedGradient",
    "FixedFlux",
    "FixedLaplacian",
    "Laplace",
]

from pina._src.equation.equation_interface import EquationInterface
from pina._src.equation.base_equation import BaseEquation
from pina._src.equation.equation import Equation
from pina._src.equation.system_equation import SystemEquation
from pina._src.equation.equation_factory import (
    FixedFlux,
    FixedGradient,
    FixedLaplacian,
    FixedValue,
    Laplace,
)
