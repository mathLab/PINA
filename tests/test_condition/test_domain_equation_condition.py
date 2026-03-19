import pytest
import torch
from pina import Condition
from pina import LabelTensor
from pina.domain import CartesianDomain
from pina.equation.zoo import FixedValue
from pina.condition import DomainEquationCondition


# Define a simple domain and equation for testing
domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
equation = FixedValue(0.0)
from pina._src.equation.equation_factory import FixedValue
from pina.equation import Equation
from pina.condition import DomainEquationCondition


class DummySolver:
    def __init__(self):
        self._params = {"shift": torch.tensor(0.25)}

    def forward(self, samples):
        return samples.extract(["x"]) - samples.extract(["y"])

example_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
example_equation = FixedValue(0.0)


def test_constructor():

    # Define the condition
    condition = Condition(domain=domain, equation=equation)

    # Assert correct types
    assert isinstance(condition, DomainEquationCondition)

    # Assert that the domain and equation are stored correctly
    assert condition.domain is domain
    assert condition.equation is equation

    # Assert that the data attribute is set to None
    assert hasattr(condition, "data")
    assert condition.data is None

    # Should fail if domain is not an instance of DomainInterface or a string
    with pytest.raises(ValueError):
        Condition(domain=123, equation=equation)

    # Should fail if equation is not an instance of BaseEquation
    with pytest.raises(ValueError):
        Condition(domain=domain, equation=123)


def test_get_item():

    # Define the condition
    condition = Condition(domain=domain, equation=equation)

    # Should raise NotImplementedError when trying to access by index
    with pytest.raises(NotImplementedError):
        condition[0]


def test_create_batch():

    # Define the condition
    condition = Condition(domain=domain, equation=equation)

    # Should raise TypeError when trying to access condition.data since None
    with pytest.raises(TypeError):
        _ = [condition.data[i] for i in [0, 2, 4, 6]]
def test_getitem_not_implemented():
    cond = Condition(domain=example_domain, equation=FixedValue(0.0))
    with pytest.raises(NotImplementedError):
        cond[0]


def test_evaluate_domain_equation_condition():
    def equation_func(input_, output_, params_):
        return output_ + input_.extract(["y"]) - params_["shift"]

    samples = LabelTensor(torch.randn(12, 2), labels=["x", "y"])
    cond = Condition(domain=example_domain, equation=Equation(equation_func))
    solver = DummySolver()
    batch = {"input": samples}
    loss = torch.nn.MSELoss(reduction="none")

    residual = cond.evaluate(batch, solver, loss)
    expected = loss(
        samples.extract(["x"]) - solver._params["shift"],
        torch.zeros_like(samples.extract(["x"]) - solver._params["shift"]),
    )

    torch.testing.assert_close(residual, expected)
