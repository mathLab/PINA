"""Module to define equations and systems of equations."""

__all__ = [
    "SystemEquation",
    "Equation",
    "FixedValue",
    "FixedGradient",
    "FixedFlux",
    "FixedLaplacian",
]

from .equation import Equation
from .equation_factory import (
    FixedFlux,
    FixedGradient,
    FixedLaplacian,
    FixedValue,
)
from .system_equation import SystemEquation
