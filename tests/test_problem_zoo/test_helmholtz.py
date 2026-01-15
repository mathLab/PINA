import pytest
from pina.problem.zoo import HelmholtzProblem
from pina.problem import SpatialProblem


@pytest.mark.parametrize("alpha", [1.5, 3])
def test_constructor(alpha):

    problem = HelmholtzProblem(alpha=alpha)
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    with pytest.raises(ValueError):
        HelmholtzProblem(alpha="invalid")
