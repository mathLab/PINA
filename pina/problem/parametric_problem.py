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
        >>> from pina.operator import grad
        >>> from pina.equations import Equation, FixedValue
        >>> from pina import Condition
        >>> from pina.geometry import CartesianDomain
        >>> import torch
        >>>
        >>>
        >>> class ParametricODE(SpatialProblem, ParametricProblem):
        >>>
        >>>     output_variables = ['u']
        >>>     spatial_domain = CartesianDomain({'x': [0, 1]})
        >>>     parameter_domain = CartesianDomain({'alpha': [1, 10]})
        >>>
        >>>     def ode_equation(input_, output_):
        >>>         u_x = grad(output_, input_, components=['u'], d=['x'])
        >>>         u = output_.extract(['u'])
        >>>         alpha = input_.extract(['alpha'])
        >>>         return alpha * u_x - u
        >>>
        >>>     conditions = {
        >>>         'x0': Condition(CartesianDomain({'x': 0, 'alpha':[1, 10]}), FixedValue(1.)),
        >>>         'D': Condition(CartesianDomain({'x': [0, 1], 'alpha':[1, 10]}), Equation(ode_equation))}
    """

    @abstractmethod
    def parameter_domain(self):
        """
        The parameters' domain of the problem.
        """
        pass

    @property
    def parameters(self):
        """
        The parameters' variables of the problem.
        """
        return self.parameter_domain.variables
