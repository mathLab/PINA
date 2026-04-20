from pina.equation.zoo import FixedLaplacian
from pina import LabelTensor
import torch
import pytest

# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "t"])
u = torch.pow(pts, 2)
u.labels = ["u", "v", "w"]


@pytest.mark.parametrize("value", [0, 10, -7.5])
@pytest.mark.parametrize("components", [None, "u", ["u", "w"]])
@pytest.mark.parametrize("d", [None, "x", ["x", "y"]])
def test_fixed_laplacian(value, components, d):

    # Constructor
    equation = FixedLaplacian(value=value, components=components, d=d)

    # Residual
    residual = equation.residual(pts, u)
    len_c = len(components) if components is not None else u.shape[1]
    assert residual.shape == (pts.shape[0], len_c)
