from pina.problem.zoo import InversePoisson2DSquareProblem
from pina.problem import InverseProblem, SpatialProblem


def test_constructor():
    problem = InversePoisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, InverseProblem)
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
