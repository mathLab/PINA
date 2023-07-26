""" Stokes flow problem"""
import torch

from pina.problem import SpatialProblem
from pina.operators import laplacian, grad, div
from pina import Condition, CartesianDomain, LabelTensor


class Stokes(SpatialProblem):
    """ Stokes flow problem"""

    output_variables = ['ux', 'uy', 'p']
    spatial_domain = CartesianDomain({'x': [-2, 2], 'y': [-1, 1]})

    def momentum(self, input_, output_):
        """ Momentum equation."""
        delta_ = torch.hstack((LabelTensor(laplacian(output_.extract(['ux']), input_), ['x']),
                               LabelTensor(laplacian(output_.extract(['uy']), input_), ['y'])))
        return - delta_ + grad(output_.extract(['p']), input_)

    def continuity(self, input_, output_):
        """ Continuity equation."""
        return div(output_.extract(['ux', 'uy']), input_)

    def inlet(self, input_, output_):
        """ Inlet boundary condition."""
        value = 2 * (1 - input_.extract(['y'])**2)
        return output_.extract(['ux']) - value

    def outlet(self, input_, output_):
        """ Outlet boundary condition."""
        value = 0.0
        return output_.extract(['p']) - value

    def wall(self, input_, output_):
        """ Wall boundary condition."""
        value = 0.0
        return output_.extract(['ux', 'uy']) - value

    conditions = {
        'gamma_top': Condition(CartesianDomain({'x': [-2, 2], 'y':  1}),
                               wall),
        'gamma_bot': Condition(CartesianDomain({'x': [-2, 2], 'y': -1}),
                               wall),
        'gamma_out': Condition(CartesianDomain({'x':  2, 'y': [-1, 1]}),
                               outlet),
        'gamma_in':  Condition(CartesianDomain({'x': -2, 'y': [-1, 1]}),
                               inlet),
        'D': Condition(CartesianDomain({'x': [-2, 2], 'y': [-1, 1]}),
                       [momentum, continuity]),
    }
