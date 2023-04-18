"""Module for the TimeDependentProblem class"""
from abc import abstractmethod

from .abstract_problem import AbstractProblem


class TimeDependentProblem(AbstractProblem):
    """
    The class for the definition of time-dependent problems, i.e., for problems
    depending on time.

    Here's an example of a 1D wave problem.

    :Example:
        >>> from pina.problem import SpatialProblem, TimeDependentProblem
        >>> from pina.operators import grad, nabla
        >>> from pina import Condition, Span
        >>> import torch
        >>>
        >>> class Wave(TimeDependentSpatialProblem):
        >>>
        >>>     output_variables = ['u']
        >>>     spatial_domain = Span({'x': [0, 3]})
        >>>     temporal_domain = Span({'t': [0, 1]})
        >>>
        >>>     def wave_equation(input_, output_):
        >>>         u_t = grad(output_, input_, components=['u'], d=['t'])
        >>>         u_tt = grad(u_t, input_, components=['dudt'], d=['t'])
        >>>         nabla_u = nabla(output_, input_, components=['u'], d=['x'])
        >>>         return nabla_u - u_tt
        >>>
        >>>     def nil_dirichlet(input_, output_):
        >>>         value = 0.0
        >>>         return output_.extract(['u']) - value
        >>>
        >>>     def initial_condition(input_, output_):
        >>>         u_expected = (-3*torch.sin(2*torch.pi*input_.extract(['x']))
        >>>             + 5*torch.sin(8/3*torch.pi*input_.extract(['x'])))
        >>>         u = output_.extract(['u'])
        >>>         return u - u_expected
        >>>
        >>>     conditions = {
        >>>         't0': Condition(Span({'x': [0, 3], 't':0}), initial_condition),
        >>>         'gamma1': Condition(Span({'x':0, 't':[0, 1]}), nil_dirichlet),
        >>>         'gamma2': Condition(Span({'x':3, 't':[0, 1]}), nil_dirichlet),
        >>>         'D': Condition(Span({'x': [0, 3], 't':[0, 1]}), wave_equation)}

    """

    @abstractmethod
    def temporal_domain(self):
        """
        The temporal domain of the problem.
        """
        pass

    @property
    def temporal_variable(self):
        """
        The time variable of the problem.
        """
        return self.temporal_domain.variables
