import pytest
import torch
from pina import LabelTensor
from pina.equation.zoo import AcousticWaveEquation


# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "t"])
u = torch.pow(pts, 2)
u.labels = ["u", "v", "w"]


@pytest.mark.parametrize("c", [1.0, 10, -7.5])
def test_acoustic_wave_equation(c):

    # Constructor
    equation = AcousticWaveEquation(c=c)

    # Should fail if c is not a float or int
    with pytest.raises(ValueError):
        AcousticWaveEquation(c="invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape

    # Should fail if the input has no 't' label
    with pytest.raises(ValueError):
        residual = equation.residual(pts["x", "y"], u)
