"""Module for the TimeDependentProblem class"""

from abc import abstractmethod

from .abstract_problem import AbstractProblem


class TimeDependentProblem(AbstractProblem):
    """
    The class for the definition of time-dependent problems, i.e., for problems
    depending on time.

    Here's an example of a 1D wave problem.

    :Example:
        TODO
    """

    @abstractmethod
    def temporal_domain(self):
        """
        The temporal domain of the problem.
        """

    @property
    def temporal_variable(self):
        """
        The time variable of the problem.
        """
        return self.temporal_domain.variables
