import numpy as np
import torch

from pina.problem import SpatialProblem
from pina.operators import nabla
from pina import Condition, Span


class Poisson(SpatialProblem):

    spatial_variables = ['x', 'y']
    output_variables = ['u']
    domain = Span({'x': [0, 1], 'y': [0, 1]})

    def laplace_equation(input_, output_):
        force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                      torch.sin(input_.extract(['y'])*torch.pi))
        nabla_u = nabla(output_.extract(['u']), input_)
        return nabla_u - force_term

    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(Span({'x': [-1, 1], 'y':  1}), nil_dirichlet),
        'gamma2': Condition(Span({'x': [-1, 1], 'y': -1}), nil_dirichlet),
        'gamma3': Condition(Span({'x':  1, 'y': [-1, 1]}), nil_dirichlet),
        'gamma4': Condition(Span({'x': -1, 'y': [-1, 1]}), nil_dirichlet),
        'D': Condition(Span({'x': [-1, 1], 'y': [-1, 1]}), laplace_equation),
    }

    def poisson_sol(self, x, y):
        return -(np.sin(x*np.pi)*np.sin(y*np.pi))/(2*np.pi**2)

    truth_solution = poisson_sol
