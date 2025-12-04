import torch
import pytest
from scipy.interpolate import BSpline
from pina.operator import grad
from pina.model import Spline
from pina import LabelTensor


# Utility quantities for testing
order = torch.randint(3, 6, (1,)).item()
n_ctrl_pts = torch.randint(order, order + 5, (1,)).item()
n_knots = order + n_ctrl_pts

# Input tensor
points = [
    LabelTensor(torch.rand(100, 1), ["x"]),
    LabelTensor(torch.rand(2, 100, 1), ["x"]),
]


# Function to compare with scipy implementation
def check_scipy_spline(model, x, output_):

    # Define scipy spline
    scipy_spline = BSpline(
        t=model.knots.detach().numpy(),
        c=model.control_points.detach().numpy(),
        k=model.order - 1,
    )

    # Compare outputs
    torch.allclose(
        output_,
        torch.tensor(scipy_spline(x), dtype=output_.dtype),
        atol=1e-5,
        rtol=1e-5,
    )


# Define all possible combinations of valid arguments for Spline class
valid_args = [
    {
        "order": order,
        "control_points": torch.rand(n_ctrl_pts),
        "knots": torch.linspace(0, 1, n_knots),
    },
    {
        "order": order,
        "control_points": torch.rand(n_ctrl_pts),
        "knots": {"n": n_knots, "min": 0, "max": 1, "mode": "auto"},
    },
    {
        "order": order,
        "control_points": torch.rand(n_ctrl_pts),
        "knots": {"n": n_knots, "min": 0, "max": 1, "mode": "uniform"},
    },
    {
        "order": order,
        "control_points": None,
        "knots": torch.linspace(0, 1, n_knots),
    },
    {
        "order": order,
        "control_points": None,
        "knots": {"n": n_knots, "min": 0, "max": 1, "mode": "auto"},
    },
    {
        "order": order,
        "control_points": None,
        "knots": {"n": n_knots, "min": 0, "max": 1, "mode": "uniform"},
    },
    {
        "order": order,
        "control_points": torch.rand(n_ctrl_pts),
        "knots": None,
    },
]


@pytest.mark.parametrize("args", valid_args)
def test_constructor(args):
    Spline(**args)

    # Should fail if order is not a positive integer
    with pytest.raises(AssertionError):
        Spline(
            order=-1, control_points=args["control_points"], knots=args["knots"]
        )

    # Should fail if control_points is not None or a torch.Tensor
    with pytest.raises(ValueError):
        Spline(
            order=args["order"], control_points=[1, 2, 3], knots=args["knots"]
        )

    # Should fail if knots is not None, a torch.Tensor, or a dict
    with pytest.raises(ValueError):
        Spline(
            order=args["order"], control_points=args["control_points"], knots=5
        )

    # Should fail if both knots and control_points are None
    with pytest.raises(ValueError):
        Spline(order=args["order"], control_points=None, knots=None)

    # Should fail if knots is not one-dimensional
    with pytest.raises(ValueError):
        Spline(
            order=args["order"],
            control_points=args["control_points"],
            knots=torch.rand(n_knots, 4),
        )

    # Should fail if control_points is not one-dimensional
    with pytest.raises(ValueError):
        Spline(
            order=args["order"],
            control_points=torch.rand(n_ctrl_pts, 4),
            knots=args["knots"],
        )

    # Should fail if the number of knots != order + number of control points
    # If control points are None, they are initialized to fulfill this condition
    if args["control_points"] is not None:
        with pytest.raises(ValueError):
            Spline(
                order=args["order"],
                control_points=args["control_points"],
                knots=torch.linspace(0, 1, n_knots + 1),
            )

    # Should fail if the knot dict is missing required keys
    with pytest.raises(ValueError):
        Spline(
            order=args["order"],
            control_points=args["control_points"],
            knots={"n": n_knots, "min": 0, "max": 1},
        )

    # Should fail if the knot dict has invalid 'mode' key
    with pytest.raises(ValueError):
        Spline(
            order=args["order"],
            control_points=args["control_points"],
            knots={"n": n_knots, "min": 0, "max": 1, "mode": "invalid"},
        )


@pytest.mark.parametrize("args", valid_args)
@pytest.mark.parametrize("pts", points)
def test_forward(args, pts):

    # Define the model
    model = Spline(**args)

    # Evaluate the model
    output_ = model(pts)
    assert output_.shape == pts.shape

    # Compare with scipy implementation only for interpolant knots (mode: auto)
    if isinstance(args["knots"], dict) and args["knots"]["mode"] == "auto":
        check_scipy_spline(model, pts, output_)


@pytest.mark.parametrize("args", valid_args)
@pytest.mark.parametrize("pts", points)
def test_backward(args, pts):

    # Define the model
    model = Spline(**args)

    # Evaluate the model
    output_ = model(pts)
    loss = torch.mean(output_)
    loss.backward()
    assert model.control_points.grad.shape == model.control_points.shape


@pytest.mark.parametrize("args", valid_args)
@pytest.mark.parametrize("pts", points)
def test_derivative(args, pts):

    # Define and evaluate the model
    model = Spline(**args)
    pts.requires_grad_(True)
    output_ = LabelTensor(model(pts), "u")

    # Compute derivatives
    first_der = model.derivative(x=pts, degree=1)
    first_der_auto = grad(output_, pts).tensor

    # Check shape and value
    assert first_der.shape == pts.shape
    assert torch.allclose(first_der, first_der_auto, atol=1e-4, rtol=1e-4)
