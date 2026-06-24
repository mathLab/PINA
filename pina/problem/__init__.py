"""Module for the Problems.

:Example:

    >>> from pina.problem import SpatialProblem
    >>> from pina.domain import CartesianDomain
    >>> class MyProblem(SpatialProblem):
    ...     output_variables = ['u']
    ...     domains = {'domain': CartesianDomain({'x': [0, 1]})}
"""

__all__ = [
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
import warnings

_DEPRECATED_IMPORTS = {"AbstractProblem": "BaseProblem"}


def __getattr__(name):
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.problem' is deprecated; use "
            f"pina.problem.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]
