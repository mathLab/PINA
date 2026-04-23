import pytest
import torch
from pina import LabelTensor
from pina.equation.zoo import BurgersEquation


# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "t"])
u = torch.sin(pts["x", "y"]) * torch.cos(pts["y", "t"])
u.labels = ["u", "v"]


@pytest.mark.parametrize("nu", [0, 1, 2.5])
def test_burgers_equation(nu):

    # Constructor
    equation = BurgersEquation(nu=nu)

    # Should fail if nu is not a float or int
    with pytest.raises(ValueError):
        BurgersEquation(nu="invalid")

    # Should fail if nu is negative
    with pytest.raises(ValueError):
        BurgersEquation(nu=-1)

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape

    # Should fail if the input has no 't' label
    with pytest.raises(ValueError):
        residual = equation.residual(pts["x", "y"], u)

    # Should fail if output and spatial dimensions do not match
    with pytest.raises(ValueError):
        residual = equation.residual(pts, u["u"])
