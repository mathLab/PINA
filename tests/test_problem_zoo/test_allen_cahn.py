import pytest
from pina.problem.zoo import AllenCahnProblem
from pina.problem import SpatialProblem, TimeDependentProblem


@pytest.mark.parametrize("alpha", [0.1, 1])
@pytest.mark.parametrize("beta", [0.1, 1])
def test_constructor(alpha, beta):

    problem = AllenCahnProblem(alpha=alpha, beta=beta)
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert isinstance(problem, TimeDependentProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    # Should fail if alpha is not a float or int
    with pytest.raises(ValueError):
        AllenCahnProblem(alpha="invalid", beta=beta)

    # Should fail if beta is not a float or int
    with pytest.raises(ValueError):
        AllenCahnProblem(alpha=alpha, beta="invalid")
