import pytest
from pina import Condition
from pina.domain import CartesianDomain
from pina.equation.zoo import FixedValue
from pina.condition import DomainEquationCondition

example_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
example_equation = FixedValue(0.0)


def test_init_domain_equation():
    cond = Condition(domain=example_domain, equation=example_equation)
    assert isinstance(cond, DomainEquationCondition)
    assert cond.domain is example_domain
    assert cond.equation is example_equation
    assert hasattr(cond, "data")
    assert cond.data is None


def test_len_not_implemented():
    cond = Condition(domain=example_domain, equation=FixedValue(0.0))
    with pytest.raises(NotImplementedError):
        len(cond)


def test_getitem_not_implemented():
    cond = Condition(domain=example_domain, equation=FixedValue(0.0))
    with pytest.raises(NotImplementedError):
        cond[0]
