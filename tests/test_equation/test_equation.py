import torch
import pytest
from pina.operator import grad, laplacian
from pina.equation import Equation
from pina import LabelTensor


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


def foo():
    pass


@pytest.mark.parametrize("equation", [eq1, eq2])
def test_constructor(equation):
    Equation(equation)

    # Should fail if the equation is not a callable function
    with pytest.raises(ValueError):
        Equation([1, 2, 4])

    # Should fail if the equation is not a callable function
    with pytest.raises(ValueError):
        Equation(foo())


@pytest.mark.parametrize("equation, last_dim", [(eq1, 2), (eq2, 1)])
def test_residual(equation, last_dim):

    # Define the equation
    eq = Equation(equation)

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
    eq_res = eq.residual(pts, u)
    assert eq_res.shape == torch.Size([n_pts, last_dim])
