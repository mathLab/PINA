"""Module for the ParametricProblem class"""
import torch
from abc import abstractmethod
from .abstract_problem import AbstractProblem


class InverseProblem(AbstractProblem):
    """
    The class for the definition of inverse problems, i.e., problems
    with unknown parameters that have to be learned during the training process
    from given data.

    Here's an example of a spatial inverse ODE problem, i.e., a spatial
    ODE problem with an unknown parameter `alpha` as coefficient of the
    derivative term.

    :Example:
        >>> from pina.problem import SpatialProblem, InverseProblem
        >>> from pina.operators import grad
        >>> from pina.equation import ParametricEquation, FixedValue
        >>> from pina import Condition
        >>> from pina.geometry import CartesianDomain
        >>> import torch
        >>>
        >>> class InverseODE(SpatialProblem, InverseProblem):
        >>>
        >>>     output_variables = ['u']
        >>>     spatial_domain = CartesianDomain({'x': [0, 1]})
        >>>     unknown_parameter_domain = CartesianDomain({'alpha': [1, 10]})
        >>>
        >>>     def ode_equation(input_, output_, params_):
        >>>         u_x = grad(output_, input_, components=['u'], d=['x'])
        >>>         u = output_.extract(['u'])
        >>>         return params_.extract(['alpha']) * u_x - u
        >>>
        >>>     def solution_data(input_, output_):
        >>>         x = input_.extract(['x'])
        >>>         solution = torch.exp(x)
        >>>         return output_ - solution
        >>>
        >>>     conditions = {
        >>>         'x0': Condition(CartesianDomain({'x': 0}), FixedValue(1.0)),
        >>>         'D': Condition(CartesianDomain({'x': [0, 1]}), ParametricEquation(ode_equation)),
        >>>         'data': Condition(CartesianDomain({'x': [0, 1]}), Equation(solution_data))
    """

    def __init__(self):
        super().__init__()
        # storing unknown_parameters for optimization
        self.unknown_parameters = {}
        for i, var in enumerate(self.unknown_variables):
            range_var = self.unknown_parameter_domain.range_[var]
            tensor_var = (
                torch.rand(1, requires_grad=True) * range_var[1]
                + range_var[0]
            )
            self.unknown_parameters[var] = torch.nn.Parameter(
                tensor_var
            )

    @abstractmethod
    def unknown_parameter_domain(self):
        """
        The parameters' domain of the problem.
        """
        pass

    @property
    def unknown_variables(self):
        """
        The parameters of the problem.
        """
        return self.unknown_parameter_domain.variables

    @property
    def unknown_parameters(self):
        """
        The parameters of the problem.
        """
        return self.__unknown_parameters

    @unknown_parameters.setter
    def unknown_parameters(self, value):
        self.__unknown_parameters = value
