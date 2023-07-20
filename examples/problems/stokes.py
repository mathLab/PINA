import numpy as np
import torch

from pina.problem import SpatialProblem
from pina.operators import laplacian, grad, div
from pina import Condition, Span, LabelTensor


class Stokes(SpatialProblem):

    output_variables = ['ux', 'uy', 'p']
    spatial_domain = Span({'x': [-2, 2], 'y': [-1, 1]})

    def momentum(input_, output_):
        delta_ = torch.hstack((LabelTensor(laplacian(output_.extract(['ux']), input_), ['x']),
            LabelTensor(laplacian(output_.extract(['uy']), input_), ['y'])))
        return - delta_ + grad(output_.extract(['p']), input_)

    def continuity(input_, output_):
        return div(output_.extract(['ux', 'uy']), input_)

    def inlet(input_, output_):
        value = 2 * (1 - input_.extract(['y'])**2)
        return output_.extract(['ux']) - value

    def outlet(input_, output_):
        value = 0.0
        return output_.extract(['p']) - value

    def wall(input_, output_):
        value = 0.0
        return output_.extract(['ux', 'uy']) - value

    conditions = {
        'gamma_top': Condition(Span({'x': [-2, 2], 'y':  1}), wall),
        'gamma_bot': Condition(Span({'x': [-2, 2], 'y': -1}), wall),
        'gamma_out': Condition(Span({'x':  2, 'y': [-1, 1]}), outlet),
        'gamma_in':  Condition(Span({'x': -2, 'y': [-1, 1]}), inlet),
        'D': Condition(Span({'x': [-2, 2], 'y': [-1, 1]}), [momentum, continuity]),
    }
