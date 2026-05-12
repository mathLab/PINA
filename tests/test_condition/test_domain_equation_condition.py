import pytest
from pina import Condition
from pina.domain import CartesianDomain
from pina.equation.zoo import FixedValue
from pina.condition import DomainEquationCondition


# Define a simple domain and equation for testing
domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
equation = FixedValue(0.0)


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


def test_evaluate():

    # Define the condition
    condition = Condition(domain=domain, equation=equation)

    # Should raise NotImplementedError when trying to evaluate the condition
    with pytest.raises(NotImplementedError):
        condition.evaluate(None, None, None)
