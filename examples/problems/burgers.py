import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span


class Burgers1D(TimeDependentProblem, SpatialProblem):

    spatial_variables = ['x']
    temporal_variable = ['t']
    output_variables = ['u']
    domain = Span({'x': [-1, 1], 't': [0, 1]})

    def burger_equation(input_, output_):
        grad_u = grad(output_['u'], input_)
        grad_x = grad_u['x']
        grad_t = grad_u['t']
        gradgrad_u_x = grad(grad_u['x'], input_)
        return (
            grad_u['t'] + output_['u']*grad_u['x'] -
            (0.01/torch.pi)*gradgrad_u_x['x']
        )

    def nil_dirichlet(input_, output_):
        u_expected = 0.0
        return output_['u'] - u_expected

    def initial_condition(input_, output_):
        u_expected = -torch.sin(torch.pi*input_['x'])
        return output_['u'] - u_expected

    conditions = {
        'gamma1': Condition(Span({'x': -1, 't': [0, 1]}), nil_dirichlet),
        'gamma2': Condition(Span({'x':  1, 't': [0, 1]}), nil_dirichlet),
        't0': Condition(Span({'x': [-1, 1], 't': 0}), initial_condition),
        'D': Condition(Span({'x': [-1, 1], 't': [0, 1]}), burger_equation),
    }
