import numpy as np
import torch
from pina.segment import Segment
from pina.cube import Cube
from pina.problem import Problem2D
from pina.operators import grad, div, nabla


class Poisson2D(Problem2D):

    input_variables = ['x', 'y']
    output_variables = ['u']
    spatial_domain = Cube([[0, 1], [0, 1]])

    def laplace_equation(input_, output_):
        force_term = (torch.sin(input_['x']*torch.pi)
                        * torch.sin(input_['y']*torch.pi))
        return nabla(output_['u'], input_).flatten() - force_term

    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_['u'] - value

    conditions = {
        'gamma1': {'location': Segment((0, 0), (1, 0)), 'func': nil_dirichlet},
        'gamma2': {'location': Segment((1, 0), (1, 1)), 'func': nil_dirichlet},
        'gamma3': {'location': Segment((1, 1), (0, 1)), 'func': nil_dirichlet},
        'gamma4': {'location': Segment((0, 1), (0, 0)), 'func': nil_dirichlet},
        'D': {'location': Cube([[0, 1], [0, 1]]), 'func': laplace_equation}
    }

    def poisson_sol(self, x, y):
        return -(np.sin(x*np.pi)*np.sin(y*np.pi))/(2*np.pi**2)

    truth_solution = poisson_sol
