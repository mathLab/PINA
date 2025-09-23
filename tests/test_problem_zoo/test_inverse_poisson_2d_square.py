import pytest
from pina.problem.zoo import InversePoisson2DSquareProblem
from pina.problem import InverseProblem, SpatialProblem


@pytest.mark.parametrize("load", [True, False])
@pytest.mark.parametrize("data_size", [0.01, 0.05])
def test_constructor(load, data_size):

    # Define the problem with or without loading data
    problem = InversePoisson2DSquareProblem(load=load, data_size=data_size)

    # Discretise the domain
    problem.discretise_domain(n=10, mode="random", domains="all")

    # Check if the problem is correctly set up
    assert problem.are_all_domains_discretised
    assert isinstance(problem, InverseProblem)
    assert isinstance(problem, SpatialProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)

    # Should fail if data_size is not in the range [0.0, 1.0]
    with pytest.raises(ValueError):
        problem = InversePoisson2DSquareProblem(load=load, data_size=3.0)
