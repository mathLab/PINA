import pytest
import torch
from pina.problem.zoo import DiffusionReactionProblem
from pina.problem import TimeDependentProblem, SpatialProblem


@pytest.mark.parametrize("alpha", [0.1, 1])
def test_constructor(alpha):

    problem = DiffusionReactionProblem(alpha=alpha)
    problem.discretise_domain(n=10, mode="random", domains=None)
    assert problem.are_all_domains_discretised
    assert isinstance(problem, TimeDependentProblem)
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    # Should fail if alpha is not a float or int
    with pytest.raises(ValueError):
        problem = DiffusionReactionProblem(alpha="invalid")


@pytest.mark.parametrize("alpha", [0.1, 1])
def test_solution(alpha):

    # Find the solution to the problem
    problem = DiffusionReactionProblem(alpha=alpha)
    problem.discretise_domain(n=10, mode="grid", domains=None)
    pts = problem.discretised_domains["D"]
    solution = problem.solution(pts.requires_grad_())

    # Compute the residual
    residual = problem.conditions["D"].equation.residual(pts, solution).tensor

    # Assert the residual of the PDE is close to zero
    assert torch.allclose(residual, torch.zeros_like(residual), atol=5e-5)
