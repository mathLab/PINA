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
        grad_u = grad(output_.extract(['u']), input_)
        grad_x = grad_u.extract(['x'])
        grad_t = grad_u.extract(['t'])
        gradgrad_u_x = grad(grad_u.extract(['x']), input_)
        return (
            grad_u.extract(['t']) + output_.extract(['u'])*grad_u.extract(['x']) -
            (0.01/torch.pi)*gradgrad_u_x.extract(['x'])
        )

    def nil_dirichlet(input_, output_):
        u_expected = 0.0
        return output_.extract(['u']) - u_expected

    def initial_condition(input_, output_):
        u_expected = -torch.sin(torch.pi*input_.extract(['x']))
        return output_.extract(['u']) - u_expected

    conditions = {
        'gamma1': Condition(Span({'x': -1, 't': [0, 1]}), nil_dirichlet),
        'gamma2': Condition(Span({'x':  1, 't': [0, 1]}), nil_dirichlet),
        't0': Condition(Span({'x': [-1, 1], 't': 0}), initial_condition),
        'D': Condition(Span({'x': [-1, 1], 't': [0, 1]}), burger_equation),
    }
