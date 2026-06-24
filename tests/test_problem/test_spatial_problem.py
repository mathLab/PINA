from pina.problem import SpatialProblem
from pina.domain import CartesianDomain


# Dummy spatial problem for testing
class DummySpatialProblem(SpatialProblem):

    output_variables = ["u"]
    conditions = {}

    # Define the spatial domain
    spatial_domain = CartesianDomain({"x": [-1, 1]})


def test_spatial_problem_initialization():

    # Initialize the dummy spatial problem
    problem = DummySpatialProblem()

    # Check that the spatial problem is initialized correctly
    assert problem.spatial_variables == ["x"]
    assert problem.input_variables == problem.spatial_variables
