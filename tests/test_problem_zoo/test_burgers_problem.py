import pytest
from pina.problem.zoo import BurgersProblem
from pina.problem import SpatialProblem, TimeDependentProblem


@pytest.mark.parametrize("nu", [0.1, 1])
def test_constructor(nu):

    problem = BurgersProblem(nu=nu)
    problem.discretise_domain(n=10, mode="random", domains=None)
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert isinstance(problem, TimeDependentProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    # Should fail if nu is not a float or int
    with pytest.raises(ValueError):
        BurgersProblem(nu="invalid")

    # Should fail if nu is negative
    with pytest.raises(ValueError):
        BurgersProblem(nu=-0.1)
