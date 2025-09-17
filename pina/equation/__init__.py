"""Module to define equations and systems of equations."""

__all__ = [
    "SystemEquation",
    "Equation",
    "FixedValue",
    "FixedGradient",
    "FixedFlux",
    "FixedLaplacian",
    "Laplace",
    "Advection",
    "AllenCahn",
    "DiffusionReaction",
    "Helmholtz",
    "Poisson",
]

from .equation import Equation
from .equation_factory import (
    FixedFlux,
    FixedGradient,
    FixedLaplacian,
    FixedValue,
    Laplace,
    Advection,
    AllenCahn,
    DiffusionReaction,
    Helmholtz,
    Poisson,
)
from .system_equation import SystemEquation
