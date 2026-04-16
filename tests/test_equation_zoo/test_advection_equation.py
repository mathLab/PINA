import pytest
import torch
from pina import LabelTensor
from pina.equation.zoo import AdvectionEquation


# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "t"])
u = torch.pow(pts, 2)
u.labels = ["u", "v", "w"]


@pytest.mark.parametrize("c", [1.0, 10, [1, 2.5]])
def test_advection_equation(c):

    # Constructor
    equation = AdvectionEquation(c)

    # Should fail if c is an empty list
    with pytest.raises(ValueError):
        AdvectionEquation([])

    # Should fail if c is not a float, int, or list
    with pytest.raises(ValueError):
        AdvectionEquation("invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape

    # Should fail if the input has no 't' label
    with pytest.raises(ValueError):
        residual = equation.residual(pts["x", "y"], u)

    # Should fail if c is a list and its length != spatial dimension
    with pytest.raises(ValueError):
        equation = AdvectionEquation([1, 2, 3])
        residual = equation.residual(pts, u)
