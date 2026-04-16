from pina.equation import SystemEquation, FixedValue, FixedGradient
from pina.operator import grad, laplacian
from pina import LabelTensor
import torch
import pytest


# Define equations for testing
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


def reduction_fn(residuals, dim):
    return torch.sum(residuals, dim=dim) / residuals.shape[dim]


# Test cases for the SystemEquation class
eq_list1 = [eq1, eq2]
eq_list2 = [FixedValue(value=0.0), FixedGradient(value=0.0, components=["u2"])]
eq_list3 = [FixedValue(value=0.0, components=["u1"]), eq1]


@pytest.mark.parametrize("eq_list", [eq_list1, eq_list2, eq_list3])
@pytest.mark.parametrize("reduction", [None, "mean", "sum", reduction_fn])
def test_constructor(eq_list, reduction):

    SystemEquation(list_equation=eq_list, reduction=reduction)

    # Should fail if the list of equations is not a list
    with pytest.raises(ValueError):
        SystemEquation(list_equation=eq1, reduction=reduction)

    # Should fail if any element of the list is neither callable nor Equation
    with pytest.raises(ValueError):
        SystemEquation(list_equation=[eq1, "equation"], reduction=reduction)

    # Should fail if the reduction is not available
    with pytest.raises(ValueError):
        SystemEquation(list_equation=[eq1, eq2], reduction="foo")


@pytest.mark.parametrize("reduction", [None, "mean", "sum", reduction_fn])
@pytest.mark.parametrize(
    "eq_list, last_dim",
    [(eq_list1, 3), (eq_list2, 4), (eq_list3, 3)],
)
def test_residual(eq_list, last_dim, reduction):

    # Define the system of equations
    system_eq = SystemEquation(list_equation=eq_list, reduction=reduction)

    # Manage number of points and variables
    n_pts = 10
    input_vars = ["x", "y"]
    output_vars = ["u1", "u2"]

    # Define the input and output tensors
    pts = LabelTensor(
        torch.rand(n_pts, len(input_vars), requires_grad=True),
        labels=input_vars,
    )
    u = torch.pow(pts, 2)
    u.labels = output_vars

    # Compute the residuals and check the shape
    res = system_eq.residual(pts, u)
    shape = (
        torch.Size([n_pts, last_dim])
        if reduction is None
        else torch.Size([n_pts])
    )
    assert res.shape == shape
