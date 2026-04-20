"""Module for the Problems."""

__all__ = [
    "AbstractProblem",  # back-compatibility with version 0.2, to be removed soon
    "ProblemInterface",
    "BaseProblem",
    "SpatialProblem",
    "TimeDependentProblem",
    "ParametricProblem",
    "InverseProblem",
]

from pina._src.problem.problem_interface import ProblemInterface
from pina._src.problem.base_problem import BaseProblem
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.problem.parametric_problem import ParametricProblem
from pina._src.problem.inverse_problem import InverseProblem

# Back-compatibility with version 0.2, to be removed soon
from pina._src.problem.base_problem import AbstractProblem
