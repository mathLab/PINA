import torch
from pina.problem.zoo import Poisson2DSquareProblem
from pina.problem import SpatialProblem


def test_constructor():

    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="random", domains=None)
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)


def test_solution():

    # Find the solution to the problem
    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="grid", domains=None)
    pts = problem.discretised_domains["D"]
    solution = problem.solution(pts.requires_grad_())

    # Compute the residual
    residual = problem.conditions["D"].equation.residual(pts, solution).tensor

    # Assert the residual of the PDE is close to zero
    assert torch.allclose(residual, torch.zeros_like(residual), atol=5e-5)
