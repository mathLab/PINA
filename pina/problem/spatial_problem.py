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
        >>> from pina import Condition, Span
        >>> import torch
        >>> class SimpleODE(SpatialProblem):
        >>>     output_variables = ['u']
        >>>     spatial_domain = Span({'x': [0, 1]})
        >>>     def ode_equation(input_, output_):
        >>>         u_x = grad(output_, input_, components=['u'], d=['x'])
        >>>         u = output_.extract(['u'])
        >>>         return u_x - u
        >>> 
        >>>     def initial_condition(input_, output_):
        >>>         value = 1.0
        >>>         u = output_.extract(['u'])
        >>>         return u - value
        >>>
        >>>     conditions = {
        >>>         'x0': Condition(Span({'x': 0.}), initial_condition),
        >>>         'D': Condition(Span({'x': [0, 1]}), ode_equation)}

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
