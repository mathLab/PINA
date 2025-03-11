import pytest
from pina.problem.zoo import AdvectionProblem
from pina.problem import SpatialProblem, TimeDependentProblem


@pytest.mark.parametrize("c", [1.5, 3])
def test_constructor(c):
    problem = AdvectionProblem(c=c)
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert isinstance(problem, TimeDependentProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    with pytest.raises(TypeError):
        AdvectionProblem(c="a")
