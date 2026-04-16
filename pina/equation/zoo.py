"""Module for implemented equations."""

__all__ = [
    "AdvectionEquation",
    "AllenCahnEquation",
    "DiffusionReactionEquation",
    "HelmholtzEquation",
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
