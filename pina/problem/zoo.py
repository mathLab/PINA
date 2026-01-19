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
]

from pina._src.problem.zoo.supervised_problem import SupervisedProblem
from pina._src.problem.zoo.helmholtz import HelmholtzProblem
from pina._src.problem.zoo.allen_cahn import AllenCahnProblem
from pina._src.problem.zoo.advection import AdvectionProblem
from pina._src.problem.zoo.poisson_2d_square import Poisson2DSquareProblem
from pina._src.problem.zoo.diffusion_reaction import DiffusionReactionProblem
from pina._src.problem.zoo.inverse_poisson_2d_square import (
    InversePoisson2DSquareProblem,
)
from pina._src.problem.zoo.acoustic_wave import AcousticWaveProblem
