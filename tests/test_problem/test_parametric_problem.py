from pina.problem import ParametricProblem
from pina.domain import CartesianDomain


# Dummy parametric problem for testing
class DummyParametricProblem(ParametricProblem):

    output_variables = ["u"]
    conditions = {}

    # Define the parameter domain
    parameter_domain = CartesianDomain({"mu": [-1, 1]})


def test_parametric_problem_initialization():

    # Initialize the dummy parametric problem
    problem = DummyParametricProblem()

    # Check that the parametric problem is initialized correctly
    assert problem.parameters == ["mu"]
    assert problem.input_variables == problem.parameters
