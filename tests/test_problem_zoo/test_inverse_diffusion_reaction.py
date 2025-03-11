from pina.problem.zoo import InverseDiffusionReactionProblem
from pina.problem import InverseProblem, SpatialProblem, TimeDependentProblem


def test_constructor():
    problem = InverseDiffusionReactionProblem()
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, InverseProblem)
    assert isinstance(problem, SpatialProblem)
    assert isinstance(problem, TimeDependentProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
