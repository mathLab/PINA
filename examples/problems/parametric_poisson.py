import numpy as np
import torch
from pina.segment import Segment
from pina.cube import Cube
from pina.problem2d import Problem2D
from pina.problem import Problem


class ParametricPoisson2DProblem(Problem2D):

    def __init__(self):

        def laplace_equation(input_, param_, output_):
            grad_u = self.grad(output_['u'], input_)
            gradgrad_u_x = self.grad(grad_u['x'], input_)
            gradgrad_u_y = self.grad(grad_u['y'], input_)
            force_term = torch.exp(
                    - 2*(input_['x'] - input_['mu1'])**2 -
                    2*(input_['y'] - input_['mu2'])**2
            )
            return gradgrad_u_x['x'] + gradgrad_u_y['y'] - force_term

        def nil_dirichlet(input_, param_, output_):
            value = 0.0
            return output_['u'] - value

        self.conditions = {
            'gamma1': {'location': Segment((-1, -1), ( 1, -1)),'func': nil_dirichlet},
            'gamma2': {'location': Segment(( 1, -1), ( 1,  1)),'func': nil_dirichlet},
            'gamma3': {'location': Segment(( 1,  1), (-1,  1)),'func': nil_dirichlet},
            'gamma4': {'location': Segment((-1,  1), (-1, -1)),'func': nil_dirichlet},
            'D': {'location': Cube([[-1, 1], [-1, 1]]), 'func': laplace_equation}
        }

        self.input_variables = ['x', 'y']
        self.output_variables = ['u']
        self.parameters = ['mu1', 'mu2']
        #self.truth_solution = poisson_sol
        self.spatial_domain = Cube([[0, 1], [0, 1]])
        self.parameter_domain = np.array([[-1, 1], [-1, 1]])


        #self.check() # Check the problem is correctly set
