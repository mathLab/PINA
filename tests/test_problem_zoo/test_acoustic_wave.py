import pytest
from pina.problem.zoo import AcousticWaveProblem
from pina.problem import SpatialProblem, TimeDependentProblem


@pytest.mark.parametrize("c", [0.1, 1])
def test_constructor(c):

    problem = AcousticWaveProblem(c=c)
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert isinstance(problem, TimeDependentProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    # Should fail if c is not a float or int
    with pytest.raises(ValueError):
        AcousticWaveProblem(c="invalid")
