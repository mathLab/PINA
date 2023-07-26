""" Parametric Poisson equation"""
import torch

from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import laplacian
from pina import CartesianDomain, Condition


class ParametricPoisson(SpatialProblem, ParametricProblem):
    """ Parametric Poisson equation"""

    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-1, 1], 'y': [-1, 1]})
    parameter_domain = CartesianDomain({'mu1': [-1, 1], 'mu2': [-1, 1]})

    def laplace_equation(self, input_, output_):
        """ Poisson equation."""
        force_term = torch.exp(
            - 2*(input_.extract(['x']) - input_.extract(['mu1']))**2
            - 2*(input_.extract(['y']) - input_.extract(['mu2']))**2)
        return laplacian(output_.extract(['u']), input_) - force_term

    def nil_dirichlet(self, input_, output_):
        """ Dirichlet boundary condition."""
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(
            CartesianDomain(
                {'x': [-1, 1], 'y': 1, 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            nil_dirichlet),
        'gamma2': Condition(
            CartesianDomain(
                {'x': [-1, 1], 'y': -1, 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            nil_dirichlet),
        'gamma3': Condition(
            CartesianDomain(
                {'x': 1, 'y': [-1, 1], 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            nil_dirichlet),
        'gamma4': Condition(
            CartesianDomain(
                {'x': -1, 'y': [-1, 1], 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            nil_dirichlet),
        'D': Condition(
            CartesianDomain({'x': [-1, 1], 'y': [-1, 1],
                            'mu1': [-1, 1], 'mu2': [-1, 1]}),
            laplace_equation),
    }
