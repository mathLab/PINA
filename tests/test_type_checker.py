import pytest
import logging
import math
from pina.type_checker import enforce_types


# Definition of a test function for arguments
@enforce_types
def foo_function1(a: int, b: float) -> float:
    return a + b


# Definition of a test function for return values
@enforce_types
def foo_function2(a: int, right: bool) -> float:
    if right:
        return float(a)
    else:
        return "Hello, world!"


def test_argument_type_checking():

    # Setting logging level to INFO, which should not trigger type checking
    logging.getLogger().setLevel(logging.INFO)

    # Both should work, even if the arguments are not of the expected type
    assert math.isclose(foo_function1(a=1, b=2.0), 3.0)
    assert math.isclose(foo_function1(a=1, b=2), 3.0)

    # Setting logging level to DEBUG, which should trigger type checking
    logging.getLogger().setLevel(logging.DEBUG)

    # The second should fail, as the second argument is an int
    assert math.isclose(foo_function1(a=1, b=2.0), 3.0)
    with pytest.raises(TypeError):
        foo_function1(a=1, b=2)


def test_return_type_checking():

    # Setting logging level to INFO, which should not trigger type checking
    logging.getLogger().setLevel(logging.INFO)

    # Both should work, even if the return value is not of the expected type
    assert math.isclose(foo_function2(a=1, right=True), 1.0)
    assert foo_function2(a=1, right=False) == "Hello, world!"

    # Setting logging level to DEBUG, which should trigger type checking
    logging.getLogger().setLevel(logging.DEBUG)

    # The second should fail, as the return value is a string
    assert math.isclose(foo_function2(a=1, right=True), 1.0)
    with pytest.raises(TypeError):
        foo_function2(a=1, right=False)
