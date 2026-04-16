import pytest
import torch
from pina.problem.zoo import HelmholtzProblem
from pina.problem import SpatialProblem


@pytest.mark.parametrize("k", [1.5, 3])
@pytest.mark.parametrize("alpha_x", [1, 3])
@pytest.mark.parametrize("alpha_y", [1, 3])
def test_constructor(k, alpha_x, alpha_y):

    problem = HelmholtzProblem(k=k, alpha_x=alpha_x, alpha_y=alpha_y)
    problem.discretise_domain(n=10, mode="random", domains=None)
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    with pytest.raises(ValueError):
        HelmholtzProblem(k=1, alpha_x=1.5, alpha_y=1)


@pytest.mark.parametrize("k", [1.5, 3])
@pytest.mark.parametrize("alpha_x", [1, 3])
@pytest.mark.parametrize("alpha_y", [1, 3])
def test_solution(k, alpha_x, alpha_y):

    # Find the solution to the problem
    problem = HelmholtzProblem(k=k, alpha_x=alpha_x, alpha_y=alpha_y)
    problem.discretise_domain(n=10, mode="grid", domains=None)
    pts = problem.discretised_domains["D"]
    solution = problem.solution(pts.requires_grad_())

    # Compute the residual
    residual = problem.conditions["D"].equation.residual(pts, solution).tensor

    # Assert the residual of the PDE is close to zero
    assert torch.allclose(residual, torch.zeros_like(residual), atol=5e-5)
