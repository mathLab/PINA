from pina.problem.zoo import DiffusionReactionProblem
from pina.problem import TimeDependentProblem, SpatialProblem


def test_constructor():
    problem = DiffusionReactionProblem()
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, TimeDependentProblem)
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
