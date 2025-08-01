from pina.equation import SystemEquation, FixedValue, FixedGradient
from pina.operator import grad, laplacian
from pina import LabelTensor
import torch
import pytest


def eq1(input_, output_):
    u_grad = grad(output_, input_)
    u1_xx = grad(u_grad, input_, components=["du1dx"], d=["x"])
    u2_xy = grad(u_grad, input_, components=["du2dx"], d=["y"])
    return torch.hstack([u1_xx, u2_xy])


def eq2(input_, output_):
    force_term = torch.sin(input_.extract(["x"]) * torch.pi) * torch.sin(
        input_.extract(["y"]) * torch.pi
    )
    delta_u = laplacian(output_.extract(["u1"]), input_)
    return delta_u - force_term


def foo():
    pass


@pytest.mark.parametrize("reduction", [None, "mean", "sum"])
def test_constructor(reduction):

    # Constructor with callable functions
    SystemEquation([eq1, eq2], reduction=reduction)

    # Constructor with Equation instances
    SystemEquation(
        [
            FixedValue(value=0.0, components=["u1"]),
            FixedGradient(value=0.0, components=["u2"]),
        ],
        reduction=reduction,
    )

    # Constructor with mixed types
    SystemEquation(
        [
            FixedValue(value=0.0, components=["u1"]),
            eq1,
        ],
        reduction=reduction,
    )

    # Non-standard reduction not implemented
    with pytest.raises(NotImplementedError):
        SystemEquation([eq1, eq2], reduction="foo")

    # Invalid input type
    with pytest.raises(ValueError):
        SystemEquation(foo)


@pytest.mark.parametrize("reduction", [None, "mean", "sum"])
def test_residual(reduction):

    # Generate random points and output
    pts = LabelTensor(torch.rand(10, 2), labels=["x", "y"])
    pts.requires_grad = True
    u = torch.pow(pts, 2)
    u.labels = ["u1", "u2"]

    # System with callable functions
    system_eq = SystemEquation([eq1, eq2], reduction=reduction)
    res = system_eq.residual(pts, u)

    # Checks on the shape of the residual
    shape = torch.Size([10, 3]) if reduction is None else torch.Size([10])
    assert res.shape == shape

    # System with Equation instances
    system_eq = SystemEquation(
        [
            FixedValue(value=0.0, components=["u1"]),
            FixedGradient(value=0.0, components=["u2"]),
        ],
        reduction=reduction,
    )

    # Checks on the shape of the residual
    shape = torch.Size([10, 3]) if reduction is None else torch.Size([10])
    assert res.shape == shape

    # System with mixed types
    system_eq = SystemEquation(
        [
            FixedValue(value=0.0, components=["u1"]),
            eq1,
        ],
        reduction=reduction,
    )

    # Checks on the shape of the residual
    shape = torch.Size([10, 3]) if reduction is None else torch.Size([10])
    assert res.shape == shape
