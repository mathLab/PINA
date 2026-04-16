import pytest
import torch
from pina import LabelTensor
from pina.equation.zoo import DiffusionReactionEquation


# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "t"])
u = torch.pow(pts, 2)
u.labels = ["u", "v", "w"]


@pytest.mark.parametrize("alpha", [1.0, 10, -7.5])
@pytest.mark.parametrize(
    "forcing_term", [lambda x: torch.sin(x), lambda x: torch.exp(x)]
)
def test_diffusion_reaction_equation(alpha, forcing_term):

    # Constructor
    equation = DiffusionReactionEquation(alpha=alpha, forcing_term=forcing_term)

    # Should fail if alpha is not a float or int
    with pytest.raises(ValueError):
        DiffusionReactionEquation(alpha="invalid", forcing_term=forcing_term)

    # Should fail if forcing_term is not a callable
    with pytest.raises(ValueError):
        DiffusionReactionEquation(alpha=alpha, forcing_term="invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape

    # Should fail if the input has no 't' label
    with pytest.raises(ValueError):
        residual = equation.residual(pts["x", "y"], u)
