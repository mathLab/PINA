"""Mathematical equations and physical laws.

This module provides a framework for defining differential equations,
boundary conditions, and complex systems of equations. It includes
pre-defined physical models such as Poisson, Laplace, and Wave equations,
along with factories for common derivative-based constraints.
"""

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
    "AcousticWave",
]

from pina._src.equation.equation import Equation
from pina._src.equation.equation_factory import (
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
    AcousticWave,
)
from pina._src.equation.system_equation import SystemEquation
