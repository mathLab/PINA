import numpy as np
import torch

from pina.problem import SpatialProblem
from pina.operators import nabla, grad, div
from pina import Condition, Span, LabelTensor


class Stokes(SpatialProblem):

    spatial_variables = ['x', 'y']
    output_variables = ['ux', 'uy', 'p']
    domain = Span({'x': [-2, 2], 'y': [-1, 1]})

    def momentum(input_, output_):
        #print(nabla(output_['ux', 'uy'], input_))
        #print(grad(output_['p'], input_))
        nabla_ = LabelTensor.hstack([
            LabelTensor(nabla(output_['ux'], input_), ['x']),
            LabelTensor(nabla(output_['uy'], input_), ['y'])])
        #return LabelTensor(nabla_.tensor + grad(output_['p'], input_).tensor, ['x', 'y'])
        return nabla_.tensor + grad(output_['p'], input_).tensor

    def continuity(input_, output_):
        return div(output_['ux', 'uy'], input_)

    def inlet(input_, output_):
        value = 2.0
        return output_['ux'] - value

    def outlet(input_, output_):
        value = 0.0
        return output_['p'] - value

    def wall(input_, output_):
        value = 0.0
        return output_['ux', 'uy'].tensor - value

    conditions = {
        'gamma_top': Condition(Span({'x': [-2, 2], 'y':  1}), wall),
        'gamma_bot': Condition(Span({'x': [-2, 2], 'y': -1}), wall),
        'gamma_out': Condition(Span({'x':  2, 'y': [-1, 1]}), outlet),
        'gamma_in':  Condition(Span({'x': -2, 'y': [-1, 1]}), inlet),
        'D': Condition(Span({'x': [-2, 2], 'y': [-1, 1]}), [momentum, continuity]),
    }
