"""Module for the ParametricProblem class."""

from abc import abstractmethod
from pina._src.problem.abstract_problem import AbstractProblem


class ParametricProblem(AbstractProblem):
    """
    Base class for all parametric problems, extending the standard problem
    definition with parameter-dependent inputs.

    A parametric problem includes additional input variables, defined over a
    dedicated parameter domain, which represent external quantities
    (e.g., physical coefficients or control variables) that can vary across
    different evaluations and influence the solution.

    This class is not meant to be instantiated directly.
    """

    @property
    @abstractmethod
    def parameter_domain(self):
        """
        The domain of the parameters of the problem.
        """

    @property
    def parameters(self):
        """
        The parameters of the problem.

        :return: The parameters of the problem.
        :rtype: list[str]
        """
        return self.parameter_domain.variables
