from pina.equation.zoo import FixedValue
from pina import LabelTensor
import torch
import pytest

# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "t"])
u = torch.pow(pts, 2)
u.labels = ["u", "v", "w"]


@pytest.mark.parametrize("value", [0, 10, -7.5])
@pytest.mark.parametrize("components", [None, "u", ["u", "w"]])
def test_fixed_value(value, components):

    # Constructor
    equation = FixedValue(value=value, components=components)

    # Residual
    residual = equation.residual(pts, u)
    len_c = len(components) if components is not None else u.shape[1]
    assert residual.shape == (pts.shape[0], len_c)

    # Should fail if value is neither a float nor an integer
    with pytest.raises(ValueError):
        FixedValue(value="not a number", components=components)

    # Should fail if components is neither a string nor a list of strings
    with pytest.raises(ValueError):
        FixedValue(value=value, components=123)
