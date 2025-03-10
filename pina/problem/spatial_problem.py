"""Module for the SpatialProblem class"""

from abc import abstractmethod

from .abstract_problem import AbstractProblem


class SpatialProblem(AbstractProblem):
    """
    The class for the definition of spatial problems, i.e., for problems
    with spatial input variables.

    Here's an example of a spatial 1-dimensional ODE problem.

    :Example:
        TODO
    """

    @abstractmethod
    def spatial_domain(self):
        """
        The spatial domain of the problem.
        """

    @property
    def spatial_variables(self):
        """
        The spatial input variables of the problem.
        """
        return self.spatial_domain.variables
