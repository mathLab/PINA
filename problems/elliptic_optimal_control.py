import numpy as np
import torch
from pina.problem import Problem
from pina.segment import Segment
from pina.cube import Cube
from pina.problem2d import Problem2D

xmin, xmax, ymin, ymax = -1, 1, -1, 1

class EllipticOptimalControl(Problem2D):

    def __init__(self, alpha=1):

        def term1(input_, output_):
            grad_p = self.grad(output_['p'], input_)
            gradgrad_p_x1 = self.grad(grad_p['x1'], input_)
            gradgrad_p_x2 = self.grad(grad_p['x2'], input_)
            yd = 2.0
            return output_['y'] - yd - (gradgrad_p_x1['x1'] + gradgrad_p_x2['x2'])

        def term2(input_, output_):
            grad_y = self.grad(output_['y'], input_)
            gradgrad_y_x1 = self.grad(grad_y['x1'], input_)
            gradgrad_y_x2 = self.grad(grad_y['x2'], input_)
            return - (gradgrad_y_x1['x1'] + gradgrad_y_x2['x2']) - output_['u']

        def term3(input_, output_):
            return output_['p'] - output_['u']*alpha


        def nil_dirichlet(input_, output_):
            y_value = 0.0
            p_value = 0.0
            return torch.abs(output_['y'] - y_value) + torch.abs(output_['p'] - p_value)

        self.conditions = {
            'gamma1': {'location': Segment((xmin, ymin), (xmax, ymin)), 'func': nil_dirichlet},
            'gamma2': {'location': Segment((xmax, ymin), (xmax, ymax)), 'func': nil_dirichlet},
            'gamma3': {'location': Segment((xmax, ymax), (xmin, ymax)), 'func': nil_dirichlet},
            'gamma4': {'location': Segment((xmin, ymax), (xmin, ymin)), 'func': nil_dirichlet},
            'D1': {'location': Cube([[xmin, xmax], [ymin, ymax]]), 'func': [term1, term2, term3]},
            #'D2': {'location': Cube([[0, 1], [0, 1]]), 'func': term2},
            #'D3': {'location': Cube([[0, 1], [0, 1]]), 'func': term3}
        }

        self.input_variables = ['x1', 'x2']
        self.output_variables = ['u', 'p', 'y']
        self.spatial_domain = Cube([[xmin, xmax], [xmin, xmax]])

