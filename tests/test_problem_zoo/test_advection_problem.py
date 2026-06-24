import pytest
import torch
from pina.problem.zoo import AdvectionProblem
from pina.problem import SpatialProblem, TimeDependentProblem


@pytest.mark.parametrize("c", [1.5, 3])
def test_constructor(c):

    problem = AdvectionProblem(c=c)
    problem.discretise_domain(n=10, mode="random", domains=None)
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert isinstance(problem, TimeDependentProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    # Should fail if c is not a float or int
    with pytest.raises(ValueError):
        AdvectionProblem(c="invalid")


@pytest.mark.parametrize("c", [1.5, 3])
def test_solution(c):

    # Find the solution to the problem
    problem = AdvectionProblem(c=c)
    problem.discretise_domain(n=10, mode="grid", domains=None)
    pts = problem.discretised_domains["D"]
    solution = problem.solution(pts.requires_grad_())

    # Compute the residual
    residual = problem.conditions["D"].equation.residual(pts, solution).tensor

    # Assert the residual of the PDE is close to zero
    assert torch.allclose(residual, torch.zeros_like(residual), atol=5e-5)
