import torch
import random
import pytest
from pina.model import SplineSurface
from pina import LabelTensor


# Utility quantities for testing
orders = [random.randint(1, 8) for _ in range(2)]
n_ctrl_pts = random.randint(max(orders), max(orders) + 5)
n_knots = [orders[i] + n_ctrl_pts for i in range(2)]

# Input tensor
points = [
    LabelTensor(torch.rand(100, 2), ["x", "y"]),
    LabelTensor(torch.rand(2, 100, 2), ["x", "y"]),
]


@pytest.mark.parametrize(
    "knots_u",
    [
        torch.rand(n_knots[0]),
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "uniform"},
        None,
    ],
)
@pytest.mark.parametrize(
    "knots_v",
    [
        torch.rand(n_knots[1]),
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "uniform"},
        None,
    ],
)
@pytest.mark.parametrize(
    "control_points", [torch.rand(n_ctrl_pts, n_ctrl_pts), None]
)
def test_constructor(knots_u, knots_v, control_points):

    # Skip if knots_u, knots_v, and control_points are all None
    if (knots_u is None or knots_v is None) and control_points is None:
        return

    SplineSurface(
        orders=orders,
        knots_u=knots_u,
        knots_v=knots_v,
        control_points=control_points,
    )

    # Should fail if orders is not list of two elements
    with pytest.raises(ValueError):
        SplineSurface(
            orders=[orders[0]],
            knots_u=knots_u,
            knots_v=knots_v,
            control_points=control_points,
        )

    # Should fail if both knots and control_points are None
    with pytest.raises(ValueError):
        SplineSurface(
            orders=orders,
            knots_u=None,
            knots_v=None,
            control_points=None,
        )

    # Should fail if control_points is not a torch.Tensor when provided
    with pytest.raises(ValueError):
        SplineSurface(
            orders=orders,
            knots_u=knots_u,
            knots_v=knots_v,
            control_points=[[0.0] * n_ctrl_pts] * n_ctrl_pts,
        )

    # Should fail if control_points is not of the correct shape when provided
    # It assumes that at least one among knots_u and knots_v is not None
    if knots_u is not None or knots_v is not None:
        with pytest.raises(ValueError):
            SplineSurface(
                orders=orders,
                knots_u=knots_u,
                knots_v=knots_v,
                control_points=torch.rand(n_ctrl_pts + 1, n_ctrl_pts + 1),
            )

    # Should fail if there are not enough knots_u to define the control points
    with pytest.raises(ValueError):
        SplineSurface(
            orders=orders,
            knots_u=torch.linspace(0, 1, orders[0]),
            knots_v=knots_v,
            control_points=None,
        )

    # Should fail if there are not enough knots_v to define the control points
    with pytest.raises(ValueError):
        SplineSurface(
            orders=orders,
            knots_u=knots_u,
            knots_v=torch.linspace(0, 1, orders[1]),
            control_points=None,
        )


@pytest.mark.parametrize(
    "knots_u",
    [
        torch.rand(n_knots[0]),
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "knots_v",
    [
        torch.rand(n_knots[1]),
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "control_points", [torch.rand(n_ctrl_pts, n_ctrl_pts), None]
)
@pytest.mark.parametrize("pts", points)
def test_forward(knots_u, knots_v, control_points, pts):

    # Define the model
    model = SplineSurface(
        orders=orders,
        knots_u=knots_u,
        knots_v=knots_v,
        control_points=control_points,
    )

    # Evaluate the model
    output_ = model(pts)
    assert output_.shape == (*pts.shape[:-1], 1)


@pytest.mark.parametrize(
    "knots_u",
    [
        torch.rand(n_knots[0]),
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "knots_v",
    [
        torch.rand(n_knots[1]),
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "control_points", [torch.rand(n_ctrl_pts, n_ctrl_pts), None]
)
@pytest.mark.parametrize("pts", points)
def test_backward(knots_u, knots_v, control_points, pts):

    # Define the model
    model = SplineSurface(
        orders=orders,
        knots_u=knots_u,
        knots_v=knots_v,
        control_points=control_points,
    )

    # Evaluate the model
    output_ = model(pts)
    loss = torch.mean(output_)
    loss.backward()
    assert model.control_points.grad.shape == model.control_points.shape
