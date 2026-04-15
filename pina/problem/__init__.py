"""Module for the Problems."""

__all__ = [
    "ProblemInterface",
    "AbstractProblem",
    "SpatialProblem",
    "TimeDependentProblem",
    "ParametricProblem",
    "InverseProblem",
]

from pina._src.problem.problem_interface import ProblemInterface
from pina._src.problem.abstract_problem import AbstractProblem
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.problem.parametric_problem import ParametricProblem
from pina._src.problem.inverse_problem import InverseProblem
