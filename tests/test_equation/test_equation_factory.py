from pina.equation import FixedValue, FixedGradient, FixedFlux, FixedLaplacian
from pina import LabelTensor
import torch
import pytest

# Define input and output values
pts = LabelTensor(torch.rand(10, 3, requires_grad=True), labels=["x", "y", "z"])
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


@pytest.mark.parametrize("value", [0, 10, -7.5])
@pytest.mark.parametrize("components", [None, "u", ["u", "w"]])
@pytest.mark.parametrize("d", [None, "x", ["x", "z"]])
def test_fixed_gradient(value, components, d):

    # Constructor
    equation = FixedGradient(value=value, components=components, d=d)

    # Residual
    residual = equation.residual(pts, u)
    len_c = len(components) if components is not None else u.shape[1]
    len_d = len(d) if d is not None else pts.shape[1]
    assert residual.shape == (pts.shape[0], len_c * len_d)


@pytest.mark.parametrize("value", [0, 10, -7.5])
@pytest.mark.parametrize("components", [None, "u", ["u", "w"]])
@pytest.mark.parametrize("d", [None, "x", ["x", "z"]])
def test_fixed_flux(value, components, d):

    # Divergence requires components and d to be of the same length
    len_c = len(components) if components is not None else u.shape[1]
    len_d = len(d) if d is not None else pts.shape[1]
    if len_c != len_d:
        return

    # Constructor
    equation = FixedFlux(value=value, components=components, d=d)

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == (pts.shape[0], 1)


@pytest.mark.parametrize("value", [0, 10, -7.5])
@pytest.mark.parametrize("components", [None, "u", ["u", "w"]])
@pytest.mark.parametrize("d", [None, "x", ["x", "z"]])
def test_fixed_laplacian(value, components, d):

    # Constructor
    equation = FixedLaplacian(value=value, components=components, d=d)

    # Residual
    residual = equation.residual(pts, u)
    len_c = len(components) if components is not None else u.shape[1]
    assert residual.shape == (pts.shape[0], len_c)
