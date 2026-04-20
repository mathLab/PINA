"""Module for the TimeDependentProblem class."""

from abc import abstractmethod
from pina._src.problem.base_problem import BaseProblem


class TimeDependentProblem(BaseProblem):
    """
    Base class for all time-dependent problems, extending the standard problem
    definition with time-dependent inputs.

    A time-dependent problem is defined over a temporal domain, where input
    variables represent the time at which the solution is evaluated.

    This class is not meant to be instantiated directly.
    """

    @property
    @abstractmethod
    def temporal_domain(self):
        """
        The domain of temporal variables of the problem.
        """

    @property
    def temporal_variables(self):
        """
        The temporal variables of the problem.

        :return: The temporal variables of the problem.
        :rtype: list[str]
        """
        return self.temporal_domain.variables
