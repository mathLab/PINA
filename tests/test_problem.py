import torch
import pytest

from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina import LabelTensor, Condition
from pina.geometry import CartesianDomain
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue


def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                    torch.sin(input_.extract(['y'])*torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term

my_laplace = Equation(laplace_equation)
in_ = LabelTensor(torch.tensor([[0., 1.]], requires_grad=True), ['x', 'y'])
out_ = LabelTensor(torch.tensor([[0.]], requires_grad=True), ['u'])

class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x': [0, 1], 'y':  1}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': 0}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x':  1, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': 0, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'D': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': [0, 1]}),
            equation=my_laplace),
        'data': Condition(
            input_points=in_,
            output_points=out_)
    }

    def poisson_sol(self, pts):
        return -(
            torch.sin(pts.extract(['x'])*torch.pi) *
            torch.sin(pts.extract(['y'])*torch.pi)
        )/(2*torch.pi**2)

    truth_solution = poisson_sol


# make the problem
poisson_problem = Poisson()


def test_discretise_domain():
    n = 10
    boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    poisson_problem.discretise_domain(n, 'grid', locations=boundaries)
    for b in boundaries:
        assert poisson_problem.input_pts[b].shape[0] == n
    poisson_problem.discretise_domain(n, 'random', locations=boundaries)
    for b in boundaries:
        assert poisson_problem.input_pts[b].shape[0] == n

    poisson_problem.discretise_domain(n, 'grid', locations=['D'])
    assert poisson_problem.input_pts['D'].shape[0] == n**2
    poisson_problem.discretise_domain(n, 'random', locations=['D'])
    assert poisson_problem.input_pts['D'].shape[0] == n

    poisson_problem.discretise_domain(n, 'latin', locations=['D'])
    assert poisson_problem.input_pts['D'].shape[0] == n

    poisson_problem.discretise_domain(n, 'lh', locations=['D'])
    assert poisson_problem.input_pts['D'].shape[0] == n

def test_sampling_few_variables():
    n = 10
    poisson_problem.discretise_domain(n, 'grid', locations=['D'], variables=['x'])
    assert poisson_problem.input_pts['D'].shape[1] == 1
    assert poisson_problem._have_sampled_points['D'] is False  

# def test_sampling_all_args():
#     n = 10
#     poisson_problem.discretise_domain(n, 'grid', locations=['D'])

# def test_sampling_all_kwargs():
#     n = 10
#     poisson_problem.discretise_domain(n=n, mode='latin', locations=['D'])

# def test_sampling_dict():
#     n = 10
#     poisson_problem.discretise_domain(
#         {'variables': ['x', 'y'], 'mode': 'grid', 'n': n}, locations=['D'])

# def test_sampling_mixed_args_kwargs():
#     n = 10
#     with pytest.raises(ValueError):
#         poisson_problem.discretise_domain(n, mode='latin', locations=['D'])
