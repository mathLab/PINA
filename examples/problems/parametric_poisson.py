import torch

from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import nabla
from pina import Span, Condition


class ParametricPoisson(SpatialProblem, ParametricProblem):

    spatial_variables = ['x', 'y']
    parameters = ['mu1', 'mu2']
    output_variables = ['u']
    domain = Span({'x': [-1, 1], 'y': [-1, 1]})

    def laplace_equation(input_, output_):
        force_term = torch.exp(
                - 2*(input_.extract(['x']) - input_.extract(['mu1']))**2 - 2*(input_.extract(['y']) -
                                                          input_.extract(['mu2']))**2)
        return nabla(output_.extract(['u']), input_) - force_term

    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(
            Span({'x': [-1, 1], 'y': 1, 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            nil_dirichlet),
        'gamma2': Condition(
            Span({'x': [-1, 1], 'y': -1, 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            nil_dirichlet),
        'gamma3': Condition(
            Span({'x': 1, 'y': [-1, 1], 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            nil_dirichlet),
        'gamma4': Condition(
            Span({'x': -1, 'y': [-1, 1], 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            nil_dirichlet),
        'D': Condition(
            Span({'x': [-1, 1], 'y': [-1, 1], 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            laplace_equation),
    }
