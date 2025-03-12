from pina.problem.zoo import AllenCahnProblem
from pina.problem import SpatialProblem, TimeDependentProblem


def test_constructor():
    problem = AllenCahnProblem()
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert isinstance(problem, TimeDependentProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
