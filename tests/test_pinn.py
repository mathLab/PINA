import torch
import pytest

from pina import LabelTensor, Condition, Span, PINN
from pina.problem import SpatialProblem
from pina.model import FeedForward
from pina.operators import nabla


class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})

    def laplace_equation(input_, output_):
        force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                      torch.sin(input_.extract(['y'])*torch.pi))
        nabla_u = nabla(output_, input_, components=['u'], d=['x', 'y'])
        return nabla_u - force_term

    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(Span({'x': [0, 1], 'y':  1}), nil_dirichlet),
        'gamma2': Condition(Span({'x': [0, 1], 'y': 0}), nil_dirichlet),
        'gamma3': Condition(Span({'x':  1, 'y': [0, 1]}), nil_dirichlet),
        'gamma4': Condition(Span({'x': 0, 'y': [0, 1]}), nil_dirichlet),
        'D': Condition(Span({'x': [0, 1], 'y': [0, 1]}), laplace_equation),
    }

    def poisson_sol(self, pts):
        return -(
            torch.sin(pts.extract(['x'])*torch.pi)*
            torch.sin(pts.extract(['y'])*torch.pi)
        )/(2*torch.pi**2)

    truth_solution = poisson_sol

problem = Poisson()
model = FeedForward(2, 1)


def test_constructor():
    PINN(problem, model)

def test_span_pts():
    pinn = PINN(problem, model)
    n = 10
    boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    pinn.span_pts(n, 'grid', boundaries)
    for b in boundaries:
        assert pinn.input_pts[b].shape[0] == n
    pinn.span_pts(n, 'random', boundaries)
    for b in boundaries:
        assert pinn.input_pts[b].shape[0] == n

    pinn.span_pts(n, 'grid', locations=['D'])
    assert pinn.input_pts['D'].shape[0] == n**2
    pinn.span_pts(n, 'random', locations=['D'])
    assert pinn.input_pts['D'].shape[0] == n
