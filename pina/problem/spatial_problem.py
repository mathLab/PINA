"""Module for the SpatialProblem class"""
from abc import abstractmethod

from .abstract_problem import AbstractProblem


class SpatialProblem(AbstractProblem):
    """
    The class for the definition of spatial problems, i.e., for problems
    with spatial input variables.

    Here's an example of a spatial 1-dimensional ODE problem.

    :Example:
        >>> from pina.problem import SpatialProblem
        >>> from pina.operators import grad
        >>> from pina.equations import Equation, FixedValue
        >>> from pina import Condition
        >>> from pina.geometry import CartesianDomain
        >>> import torch
        >>>
        >>>
        >>> class SpatialODE(SpatialProblem:
        >>>
        >>>     output_variables = ['u']
        >>>     spatial_domain = CartesianDomain({'x': [0, 1]})
        >>>
        >>>     def ode_equation(input_, output_):
        >>>         u_x = grad(output_, input_, components=['u'], d=['x'])
        >>>         u = output_.extract(['u'])
        >>>         return u_x - u
        >>>
        >>>     conditions = {
        >>>         'x0': Condition(CartesianDomain({'x': 0, 'alpha':[1, 10]}), FixedValue(1.)),
        >>>         'D': Condition(CartesianDomain({'x': [0, 1], 'alpha':[1, 10]}), Equation(ode_equation))}

    """

    @abstractmethod
    def spatial_domain(self):
        """
        The spatial domain of the problem.
        """
        pass

    @property
    def spatial_variables(self):
        """
        The spatial input variables of the problem.
        """
        return self.spatial_domain.variables
