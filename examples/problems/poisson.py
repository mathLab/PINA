""" Poisson equation example. """
import torch

from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina import Condition, CartesianDomain


class Poisson(SpatialProblem):
    """ Poisson equation example."""

    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    def laplace_equation(self, input_, output_):
        """ Poisson equation."""
        force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                      torch.sin(input_.extract(['y'])*torch.pi))
        delta_u = laplacian(output_.extract(['u']), input_)
        return delta_u - force_term

    def nil_dirichlet(self, input_, output_):
        """ Dirichlet boundary condition."""
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(CartesianDomain({'x': [0, 1], 'y':  1}),
                            nil_dirichlet),
        'gamma2': Condition(CartesianDomain({'x': [0, 1], 'y': 0}),
                            nil_dirichlet),
        'gamma3': Condition(CartesianDomain({'x':  1, 'y': [0, 1]}),
                            nil_dirichlet),
        'gamma4': Condition(CartesianDomain({'x': 0, 'y': [0, 1]}),
                            nil_dirichlet),
        'D': Condition(CartesianDomain({'x': [0, 1], 'y': [0, 1]}),
                       laplace_equation),
    }

    def poisson_sol(self, pts):
        """ Analytical solution."""
        return -(
            torch.sin(pts.extract(['x'])*torch.pi) *
            torch.sin(pts.extract(['y'])*torch.pi)
        )/(2*torch.pi**2)
        # return -(np.sin(x*np.pi)*np.sin(y*np.pi))/(2*np.pi**2)

    truth_solution = poisson_sol
