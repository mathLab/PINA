from pina.equation import (
    FixedValue,
    FixedGradient,
    FixedFlux,
    FixedLaplacian,
    Advection,
    AllenCahn,
    DiffusionReaction,
    Helmholtz,
    Poisson,
)
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


@pytest.mark.parametrize("value", [0, 10, -7.5])
@pytest.mark.parametrize("components", [None, "u", ["u", "w"]])
@pytest.mark.parametrize("d", [None, "x", ["x", "y"]])
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
@pytest.mark.parametrize("d", [None, "x", ["x", "y"]])
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
@pytest.mark.parametrize("d", [None, "x", ["x", "y"]])
def test_fixed_laplacian(value, components, d):

    # Constructor
    equation = FixedLaplacian(value=value, components=components, d=d)

    # Residual
    residual = equation.residual(pts, u)
    len_c = len(components) if components is not None else u.shape[1]
    assert residual.shape == (pts.shape[0], len_c)


@pytest.mark.parametrize("c", [1.0, 10, [1, 2.5]])
def test_advection_equation(c):

    # Constructor
    equation = Advection(c)

    # Should fail if c is an empty list
    with pytest.raises(ValueError):
        Advection([])

    # Should fail if c is not a float, int, or list
    with pytest.raises(ValueError):
        Advection("invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape

    # Should fail if the input has no 't' label
    with pytest.raises(ValueError):
        residual = equation.residual(pts["x", "y"], u)

    # Should fail if c is a list and its length != spatial dimension
    with pytest.raises(ValueError):
        equation = Advection([1, 2, 3])
        residual = equation.residual(pts, u)


@pytest.mark.parametrize("alpha", [1.0, 10, -7.5])
@pytest.mark.parametrize("beta", [1.0, 10, -7.5])
def test_allen_cahn_equation(alpha, beta):

    # Constructor
    equation = AllenCahn(alpha=alpha, beta=beta)

    # Should fail if alpha is not a float or int
    with pytest.raises(ValueError):
        AllenCahn(alpha="invalid", beta=beta)

    # Should fail if beta is not a float or int
    with pytest.raises(ValueError):
        AllenCahn(alpha=alpha, beta="invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape

    # Should fail if the input has no 't' label
    with pytest.raises(ValueError):
        residual = equation.residual(pts["x", "y"], u)


@pytest.mark.parametrize("alpha", [1.0, 10, -7.5])
@pytest.mark.parametrize(
    "forcing_term", [lambda x: torch.sin(x), lambda x: torch.exp(x)]
)
def test_diffusion_reaction_equation(alpha, forcing_term):

    # Constructor
    equation = DiffusionReaction(alpha=alpha, forcing_term=forcing_term)

    # Should fail if alpha is not a float or int
    with pytest.raises(ValueError):
        DiffusionReaction(alpha="invalid", forcing_term=forcing_term)

    # Should fail if forcing_term is not a callable
    with pytest.raises(ValueError):
        DiffusionReaction(alpha=alpha, forcing_term="invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape

    # Should fail if the input has no 't' label
    with pytest.raises(ValueError):
        residual = equation.residual(pts["x", "y"], u)


@pytest.mark.parametrize("k", [1.0, 10, -7.5])
@pytest.mark.parametrize(
    "forcing_term", [lambda x: torch.sin(x), lambda x: torch.exp(x)]
)
def test_helmholtz_equation(k, forcing_term):

    # Constructor
    equation = Helmholtz(k=k, forcing_term=forcing_term)

    # Should fail if k is not a float or int
    with pytest.raises(ValueError):
        Helmholtz(k="invalid", forcing_term=forcing_term)

    # Should fail if forcing_term is not a callable
    with pytest.raises(ValueError):
        Helmholtz(k=k, forcing_term="invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape


@pytest.mark.parametrize(
    "forcing_term", [lambda x: torch.sin(x), lambda x: torch.exp(x)]
)
def test_poisson_equation(forcing_term):

    # Constructor
    equation = Poisson(forcing_term=forcing_term)

    # Should fail if forcing_term is not a callable
    with pytest.raises(ValueError):
        Poisson(forcing_term="invalid")

    # Residual
    residual = equation.residual(pts, u)
    assert residual.shape == u.shape
