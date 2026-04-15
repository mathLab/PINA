import pytest
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
