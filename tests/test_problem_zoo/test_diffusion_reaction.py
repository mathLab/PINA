import pytest
from pina.problem.zoo import DiffusionReactionProblem
from pina.problem import TimeDependentProblem, SpatialProblem


@pytest.mark.parametrize("alpha", [0.1, 1])
def test_constructor(alpha):

    problem = DiffusionReactionProblem(alpha=alpha)
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, TimeDependentProblem)
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    # Should fail if alpha is not a float or int
    with pytest.raises(ValueError):
        problem = DiffusionReactionProblem(alpha="invalid")
