import pytest
import torch
from pina import LabelTensor
from pina.equation.zoo import AllenCahnEquation


# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "t"])
u = torch.pow(pts, 2)
u.labels = ["u", "v", "w"]


@pytest.mark.parametrize("alpha", [1.0, 10, -7.5])
@pytest.mark.parametrize("beta", [1.0, 10, -7.5])
def test_allen_cahn_equation(alpha, beta):

    # Constructor
    equation = AllenCahnEquation(alpha=alpha, beta=beta)

    # Should fail if alpha is not a float or int
    with pytest.raises(ValueError):
        AllenCahnEquation(alpha="invalid", beta=beta)

    # Should fail if beta is not a float or int
    with pytest.raises(ValueError):
        AllenCahnEquation(alpha=alpha, beta="invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape

    # Should fail if the input has no 't' label
    with pytest.raises(ValueError):
        residual = equation.residual(pts["x", "y"], u)
