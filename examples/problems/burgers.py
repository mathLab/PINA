""" Burgers equation in 1D"""
import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.geometry.cartesian import CartesianDomain


class Burgers1D(TimeDependentProblem, SpatialProblem):
    """ Burgers equation in 1D"""

    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-1, 1]})
    temporal_domain = CartesianDomain({'t': [0, 1]})

    def burger_equation(self, input_, output_):
        """ Burgers equation."""
        du = grad(output_, input_)
        ddu = grad(du, input_, components=['dudx'])
        return (
            du.extract(['dudt']) +
            output_.extract(['u'])*du.extract(['dudx']) -
            (0.01/torch.pi)*ddu.extract(['ddudxdx'])
        )

    def nil_dirichlet(self, input_, output_):
        """ Dirichlet boundary condition."""
        u_expected = 0.0
        return output_.extract(['u']) - u_expected

    def initial_condition(self, input_, output_):
        """ Initial condition."""
        u_expected = -torch.sin(torch.pi*input_.extract(['x']))
        return output_.extract(['u']) - u_expected

    conditions = {
        'gamma1': Condition(CartesianDomain({'x': -1, 't': [0, 1]}), nil_dirichlet),
        'gamma2': Condition(CartesianDomain({'x':  1, 't': [0, 1]}), nil_dirichlet),
        't0': Condition(CartesianDomain({'x': [-1, 1], 't': 0}), initial_condition),
        'D': Condition(CartesianDomain({'x': [-1, 1], 't': [0, 1]}), burger_equation),
    }
