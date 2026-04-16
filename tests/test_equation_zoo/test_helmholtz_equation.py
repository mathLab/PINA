import pytest
import torch
from pina import LabelTensor
from pina.equation.zoo import HelmholtzEquation


# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "t"])
u = torch.pow(pts, 2)
u.labels = ["u", "v", "w"]


@pytest.mark.parametrize("k", [1.0, 10, -7.5])
@pytest.mark.parametrize(
    "forcing_term", [lambda x: torch.sin(x), lambda x: torch.exp(x)]
)
def test_helmholtz_equation(k, forcing_term):

    # Constructor
    equation = HelmholtzEquation(k=k, forcing_term=forcing_term)

    # Should fail if k is not a float or int
    with pytest.raises(ValueError):
        HelmholtzEquation(k="invalid", forcing_term=forcing_term)

    # Should fail if forcing_term is not a callable
    with pytest.raises(ValueError):
        HelmholtzEquation(k=k, forcing_term="invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape
