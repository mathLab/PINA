from pina.problem import TimeDependentProblem
from pina.domain import CartesianDomain


# Dummy time-dependent problem for testing
class DummyTimeDependentProblem(TimeDependentProblem):

    output_variables = ["u"]
    conditions = {}

    # Define the temporal domain
    temporal_domain = CartesianDomain({"t": [0, 1]})


def test_time_dependent_problem_initialization():

    # Initialize the dummy time-dependent problem
    problem = DummyTimeDependentProblem()

    # Check that the time-dependent problem is initialized correctly
    assert problem.temporal_variables == ["t"]
    assert problem.input_variables == problem.temporal_variables
