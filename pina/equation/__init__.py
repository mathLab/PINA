"""Module to define equations and systems of equations."""

__all__ = [
    "SystemEquation",
    "Equation",
    "FixedValue",
    "FixedGradient",
    "FixedFlux",
    "Laplace",
]

from .equation import Equation
from .equation_factory import FixedFlux, FixedGradient, Laplace, FixedValue
from .system_equation import SystemEquation
