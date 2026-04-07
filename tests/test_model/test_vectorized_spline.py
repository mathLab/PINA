import torch
import pytest
from pina.model import VectorizedSpline, Spline
from pina.operator import grad
from pina import LabelTensor


# Utility quantities for testing
order = torch.randint(3, 6, (1,)).item()
n_ctrl_pts = torch.randint(order, order + 5, (1,)).item()
n_knots = order + n_ctrl_pts
n_splines = torch.randint(2, 5, (1,)).item()
output_dim = torch.randint(1, 4, (1,)).item()

# Input points
labels = [f"x{i}" for i in range(n_splines)]
pts = torch.rand(10, n_splines).requires_grad_(True)
pts = LabelTensor(pts, labels)


# Define all possible combinations of valid arguments for VectorizedSpline class
valid_args = [
    {
        "order": order,
        "control_points": torch.rand(n_splines, output_dim, n_ctrl_pts),
        "knots": torch.linspace(0, 1, n_knots)
        .unsqueeze(0)
        .repeat(n_splines, 1),
    },
    {
        "order": order,
        "control_points": torch.rand(n_splines, output_dim, n_ctrl_pts),
        "knots": {
            "n": n_knots,
            "min": 0,
            "max": 1,
            "mode": "auto",
            "n_splines": n_splines,
        },
    },
    {
        "order": order,
        "control_points": torch.rand(n_splines, output_dim, n_ctrl_pts),
        "knots": {
            "n": n_knots,
            "min": 0,
            "max": 1,
            "mode": "uniform",
            "n_splines": n_splines,
        },
    },
    {
        "order": order,
        "control_points": None,
        "knots": torch.linspace(0, 1, n_knots)
        .unsqueeze(0)
        .repeat(n_splines, 1),
    },
    {
        "order": order,
        "control_points": None,
        "knots": {
            "n": n_knots,
            "min": 0,
            "max": 1,
            "mode": "auto",
            "n_splines": n_splines,
        },
    },
    {
        "order": order,
        "control_points": None,
        "knots": {
            "n": n_knots,
            "min": 0,
            "max": 1,
            "mode": "uniform",
            "n_splines": n_splines,
        },
    },
    {
        "order": order,
        "control_points": torch.rand(n_splines, output_dim, n_ctrl_pts),
        "knots": None,
    },
]


@pytest.mark.parametrize("args", valid_args)
@pytest.mark.parametrize("aggregate_output", ["mean", "sum", None])
def test_constructor(args, aggregate_output):
    VectorizedSpline(**args, aggregate_output=aggregate_output)

    # Should fail if order is not a positive integer
    with pytest.raises(AssertionError):
        VectorizedSpline(
            order=-1,
            control_points=args["control_points"],
            knots=args["knots"],
            aggregate_output=aggregate_output,
        )

    # Should fail if control_points is not None or a torch.Tensor
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=[1, 2, 3],
            knots=args["knots"],
            aggregate_output=aggregate_output,
        )

    # Should fail if knots is not None, a torch.Tensor, or a dict
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots=5,
            aggregate_output=aggregate_output,
        )

    # Should fail if aggregate_output is not None, "mean", or "sum"
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots=args["knots"],
            aggregate_output="invalid",
        )

    # Should fail if both knots and control_points are None
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=None,
            knots=None,
            aggregate_output=aggregate_output,
        )

    # Should fail if knots is not two-dimensional
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots=torch.rand(n_knots),
            aggregate_output=aggregate_output,
        )

    # Should fail if control_points is not three-dimensional
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=torch.rand(n_ctrl_pts),
            knots=args["knots"],
            aggregate_output=aggregate_output,
        )

    # Should fail if the number of knots != order + number of control points
    # If control points are None, they are initialized to fulfill this condition
    if args["control_points"] is not None:
        with pytest.raises(ValueError):
            VectorizedSpline(
                order=args["order"],
                control_points=args["control_points"],
                knots=torch.linspace(0, 1, n_knots + 1)
                .unsqueeze(0)
                .repeat(n_splines, 1),
                aggregate_output=aggregate_output,
            )

    # Should fail if the knot dict is missing required keys
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots={"n": n_knots, "min": 0, "max": 1},
            aggregate_output=aggregate_output,
        )

    # Should fail if the knot dict has invalid 'mode' key
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots={"n": n_knots, "min": 0, "max": 1, "mode": "invalid"},
            aggregate_output=aggregate_output,
        )

    # Should fail if knots and control points have different number of splines
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=3,
            control_points=torch.rand(5, 4, 5),
            knots=torch.linspace(0, 1, 8).unsqueeze(0).repeat(3, 1),
            aggregate_output=aggregate_output,
        )


@pytest.mark.parametrize("args", valid_args)
def test_forward(args):

    # Define the model
    model = VectorizedSpline(**args)

    # Evaluate the model
    output_ = model(pts)

    # Check output shape
    if model.aggregate_output is None:
        assert output_.shape == (
            pts.shape[0],
            pts.shape[1],
            model.control_points.shape[1],
        )
    else:
        assert output_.shape == pts.shape


@pytest.mark.parametrize("args", valid_args)
def test_backward(args):

    # Define the model
    model = VectorizedSpline(**args)

    # Evaluate the model
    output_ = model(pts)
    loss = torch.mean(output_)
    loss.backward()
    assert model.control_points.grad.shape == model.control_points.shape


@pytest.mark.parametrize("args", valid_args)
def test_derivative(args):

    # Define and evaluate the model
    model = VectorizedSpline(**args)
    pts.requires_grad_(True)
    output_ = model(pts)

    # Compute analytical derivatives
    first_der = model.derivative(x=pts, degree=1)

    # Compute autograd derivatives -- we need to loop over output dimensions
    # since autograd doesn't support vectorized outputs
    gradients = []
    for j in range(output_.shape[2]):
        out = output_[:, :, j].squeeze(-1)
        out = LabelTensor(out, [f"u{j}" for j in range(out.shape[1])])
        gradients.append(
            grad(out, pts)[[f"du{j}dx{j}" for j in range(pts.shape[1])]]
        )
    first_der_auto = torch.stack(gradients, dim=-1)

    # Check shape and value
    assert first_der.shape == first_der_auto.shape
    assert torch.allclose(first_der, first_der_auto, atol=1e-4, rtol=1e-4)


def test_1d_vs_vectorized():

    control_points = torch.rand(1, 1, n_ctrl_pts)
    knots = torch.linspace(0, 1, n_knots).unsqueeze(0)

    # Classical 1D spline

    spline = Spline(
        order=order,
        control_points=control_points.squeeze(),
        knots=knots.squeeze(),
    )

    # Create a VectorizedSpline instance with the same control pts and knots
    vectorized_spline = VectorizedSpline(
        order=order,
        knots=knots,
        control_points=control_points,
        aggregate_output=None,
    )

    # Input points
    x = LabelTensor(torch.rand(10, 1), labels=["x"])

    # Evaluate both models on the same input
    out_spline = spline(x)
    out_vectorized = vectorized_spline(x)

    assert out_vectorized.shape == out_spline.shape
    assert torch.allclose(out_vectorized, out_spline, atol=1e-5, rtol=1e-5)
