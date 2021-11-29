import numpy as np
import torch
from pina.segment import Segment
from pina.cube import Cube
from pina.problem2d import Problem2D
from pina.problem import Problem


class Poisson2DProblem(Problem2D):

    def __init__(self):

        def laplace_equation(input_, output_):
            grad_u = self.grad(output_['u'], input_)
            gradgrad_u_x = self.grad(grad_u['x'], input_)
            gradgrad_u_y = self.grad(grad_u['y'], input_)
            #force_term = (torch.sin(input_['x']*torch.pi)
            #             * torch.sin(input_['y']*torch.pi))
            force_term = -2*(input_['y']*(1-input_['y']) +
                             input_['x']*(1-input_['x']))
            return gradgrad_u_x['x'] + gradgrad_u_y['y'] - force_term

        def nil_dirichlet(input_, output_):
            value = 0.0
            return output_['u'] - value

        self.conditions = {
            'gamma1': {'location': Segment((0, 0), (1, 0)), 'func': nil_dirichlet},
            'gamma2': {'location': Segment((1, 0), (1, 1)), 'func': nil_dirichlet},
            'gamma3': {'location': Segment((1, 1), (0, 1)), 'func': nil_dirichlet},
            'gamma4': {'location': Segment((0, 1), (0, 0)), 'func': nil_dirichlet},
            'D': {'location': Cube([[0, 1], [0, 1]]), 'func': laplace_equation}
        }

        def poisson_sol(x, y):
            return x*(1-x)*y*(1-y)

        self.input_variables = ['x', 'y']
        self.output_variables = ['u']
        self.truth_solution = poisson_sol
        self.spatial_domain = Cube([[0, 1], [0, 1]])

        #self.check() # Check the problem is correctly set
