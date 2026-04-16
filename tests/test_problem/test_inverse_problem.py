import torch
from pina.problem import InverseProblem
from pina.domain import CartesianDomain


# Dummy inverse problem for testing
class DummyInverseProblem(InverseProblem):

    output_variables = ["u"]
    conditions = {}

    # Define the unknown parameter domain
    unknown_parameter_domain = CartesianDomain({"mu": [-1, 1]})


def test_inverse_problem_initialization():

    # Initialize the dummy inverse problem
    problem = DummyInverseProblem()

    # Check that the inverse problem is initialized correctly
    assert problem.unknown_variables == ["mu"]
    assert isinstance(problem.unknown_parameters, dict)
    for k, v in problem.unknown_parameters.items():
        assert isinstance(v, torch.nn.Parameter)
        range_low, range_high = problem.unknown_parameter_domain._range[k]
        assert range_low <= v.item() <= range_high
