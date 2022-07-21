import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span


class Burgers1D(TimeDependentProblem, SpatialProblem):

    output_variables = ['u']
    spatial_domain = Span({'x': [-1, 1]})
    temporal_domain = Span({'t': [0, 1]})

    def burger_equation(input_, output_):
        du = grad(output_, input_)
        ddu = grad(du, input_, components=['dudx'])
        return (
            du.extract(['dudt']) +
            output_.extract(['u'])*du.extract(['dudx']) -
            (0.01/torch.pi)*ddu.extract(['ddudxdx'])
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
