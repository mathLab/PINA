"""Module for implemented equations."""

__all__ = [
    "AdvectionEquation",
    "AllenCahnEquation",
    "DiffusionReactionEquation",
    "FixedFlux",
    "FixedGradient",
    "FixedLaplacian",
    "FixedValue",
    "HelmholtzEquation",
    "Laplace",
    "PoissonEquation",
    "AcousticWaveEquation",
]

from pina._src.equation.zoo.acoustic_wave_equation import AcousticWaveEquation
from pina._src.equation.zoo.advection_equation import AdvectionEquation
from pina._src.equation.zoo.allen_cahn_equation import AllenCahnEquation
from pina._src.equation.zoo.diffusion_reaction_equation import (
    DiffusionReactionEquation,
)
from pina._src.equation.zoo.helmholtz_equation import HelmholtzEquation
from pina._src.equation.zoo.poisson_equation import PoissonEquation
from pina._src.equation.zoo.fixed_value import FixedValue
from pina._src.equation.zoo.fixed_gradient import FixedGradient
from pina._src.equation.zoo.fixed_flux import FixedFlux
from pina._src.equation.zoo.fixed_laplacian import FixedLaplacian, Laplace
