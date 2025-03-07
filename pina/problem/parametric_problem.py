"""Module for the ParametricProblem class"""

from abc import abstractmethod

from .abstract_problem import AbstractProblem


class ParametricProblem(AbstractProblem):
    """
    The class for the definition of parametric problems, i.e., problems
    with parameters among the input variables.

    Here's an example of a spatial parametric ODE problem, i.e., a spatial
    ODE problem with an additional parameter `alpha` as coefficient of the
    derivative term.

    :Example:
        TODO
    """

    @abstractmethod
    def parameter_domain(self):
        """
        The parameters' domain of the problem.
        """

    @property
    def parameters(self):
        """
        The parameters' variables of the problem.
        """
        return self.parameter_domain.variables
