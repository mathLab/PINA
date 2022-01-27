import numpy as np
import scipy.io
import torch

from pina.segment import Segment
from pina.cube import Cube
from pina.problem import TimeDependentProblem, Problem1D
from pina.operators import grad

def tmp_grad(output_, input_):
    return torch.autograd.grad(
            output_,
            input_.tensor,
            grad_outputs=torch.ones(output_.size()).to(
                dtype=input_.tensor.dtype,
                device=input_.tensor.device),
            create_graph=True, retain_graph=True, allow_unused=True)[0]

class Burgers1D(TimeDependentProblem, Problem1D):

    input_variables = ['x', 't']
    output_variables = ['u']
    spatial_domain = Cube([[-1, 1]])
    temporal_domain = Cube([[0, 1]])

    def burger_equation(input_, output_):
        grad_u = grad(output_['u'], input_)
        grad_x, grad_t = tmp_grad(output_['u'], input_).T
        gradgrad_u_x = grad(grad_u['x'], input_)
        grad_xx = tmp_grad(grad_x, input_)[:, 0]
        return grad_u['t'] + output_['u']*grad_u['x'] - (0.01/torch.pi)*gradgrad_u_x['x']


    def nil_dirichlet(input_, output_):
        u_expected = 0.0
        return output_['u'] - u_expected

    def initial_condition(input_, output_):
        u_expected = -torch.sin(torch.pi*input_['x'])
        return output_['u'] - u_expected



    conditions = {
        'gamma1': {'location': Segment((-1, 0), (-1, 1)), 'func': nil_dirichlet},
        'gamma2': {'location': Segment(( 1, 0), ( 1, 1)), 'func': nil_dirichlet},
        'initia': {'location': Segment((-1, 0), ( 1, 0)), 'func': initial_condition},
        'D': {'location': Cube([[-1, 1],[0,1]]), 'func': burger_equation}
    }
