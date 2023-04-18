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
        >>> from pina.problem import SpatialProblem, ParametricProblem
        >>> from pina.operators import grad
        >>> from pina import Condition, Span
        >>> import torch

        >>> class ParametricODE(SpatialProblem, ParametricProblem):

        >>>     output_variables = ['u']
        >>>     spatial_domain = Span({'x': [0, 1]})
        >>>     parameter_domain = Span({'alpha': {1, 10}})

        >>>     def ode_equation(input_, output_):
        >>>         u_x = grad(output_, input_, components=['u'], d=['x'])
        >>>         u = output_.extract(['u'])
        >>>         alpha = input_.extract(['alpha'])
        >>>         return alpha * u_x - u

        >>>     def initial_condition(input_, output_):
        >>>         value = 1.0
        >>>         u = output_.extract(['u'])
        >>>         return u - value

        >>>     conditions = {
        >>>         'x0': Condition(Span({'x': 0, 'alpha':[1, 10]}), initial_condition),
        >>>         'D': Condition(Span({'x': [0, 1], 'alpha':[1, 10]}), ode_equation)}
    """

    @abstractmethod
    def parameter_domain(self):
        pass

    @property
    def parameters(self):
        return self.parameter_domain.variables
