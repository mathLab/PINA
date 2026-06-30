"""Module for implemented problems."""

__all__ = [
    "SupervisedProblem",
    "HelmholtzProblem",
    "AllenCahnProblem",
    "AdvectionProblem",
    "Poisson2DSquareProblem",
    "DiffusionReactionProblem",
    "InversePoisson2DSquareProblem",
    "AcousticWaveProblem",
    "BurgersProblem",
]

from pina._src.problem.zoo.acoustic_wave_problem import AcousticWaveProblem
from pina._src.problem.zoo.supervised_problem import SupervisedProblem
from pina._src.problem.zoo.allen_cahn_problem import AllenCahnProblem
from pina._src.problem.zoo.advection_problem import AdvectionProblem
from pina._src.problem.zoo.helmholtz_problem import HelmholtzProblem
from pina._src.problem.zoo.poisson_problem import Poisson2DSquareProblem
from pina._src.problem.zoo.burgers_problem import BurgersProblem
from pina._src.problem.zoo.diffusion_reaction_problem import (
    DiffusionReactionProblem,
)
from pina._src.problem.zoo.inverse_poisson_problem import (
    InversePoisson2DSquareProblem,
)
