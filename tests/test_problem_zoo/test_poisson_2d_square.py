from pina.problem.zoo import Poisson2DSquareProblem
from pina.problem import SpatialProblem


def test_constructor():

    problem = Poisson2DSquareProblem()
    problem.discretise_domain(n=10, mode="random", domains="all")
    assert problem.are_all_domains_discretised
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
