import torch
import pytest

from pina.model.block import PeriodicBoundaryEmbedding, FourierFeatureEmbedding

# test tolerance
tol = 1e-6


def check_same_columns(tensor):
    # Get the first column and compute residual
    residual = tensor - tensor[0]
    zeros = torch.zeros_like(residual)
    # Compare each column with the first column
    all_same = torch.allclose(input=residual, other=zeros, atol=tol)
    return all_same


def grad(u, x):
    """
    Compute the first derivative of u with respect to x.
    """
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        allow_unused=True,
        retain_graph=True,
    )[0]


def test_constructor_PeriodicBoundaryEmbedding():
    PeriodicBoundaryEmbedding(input_dimension=1, periods=2)
    PeriodicBoundaryEmbedding(input_dimension=1, periods={"x": 3, "y": 4})
    PeriodicBoundaryEmbedding(input_dimension=1, periods={0: 3, 1: 4})
    PeriodicBoundaryEmbedding(input_dimension=1, periods=2, output_dimension=10)
    with pytest.raises(TypeError):
        PeriodicBoundaryEmbedding()
    with pytest.raises(ValueError):
        PeriodicBoundaryEmbedding(input_dimension=1.0, periods=1)
        PeriodicBoundaryEmbedding(
            input_dimension=1, periods=1, output_dimension=1.0
        )
        PeriodicBoundaryEmbedding(input_dimension=1, periods={"x": "x"})
        PeriodicBoundaryEmbedding(input_dimension=1, periods={0: "x"})


@pytest.mark.parametrize("period", [1, 4, 10])
@pytest.mark.parametrize("input_dimension", [1, 2, 3])
def test_forward_backward_same_period_PeriodicBoundaryEmbedding(
    input_dimension, period
):
    func = torch.nn.Sequential(
        PeriodicBoundaryEmbedding(
            input_dimension=input_dimension, output_dimension=60, periods=period
        ),
        torch.nn.Tanh(),
        torch.nn.Linear(60, 60),
        torch.nn.Tanh(),
        torch.nn.Linear(60, 1),
    )
    # coordinates
    x = period * torch.tensor([[0.0], [1.0]])
    if input_dimension == 2:
        x = torch.cartesian_prod(x.flatten(), x.flatten())
    elif input_dimension == 3:
        x = torch.cartesian_prod(x.flatten(), x.flatten(), x.flatten())
    x.requires_grad = True
    # output
    f = func(x)
    assert check_same_columns(f)
    # compute backward
    loss = f.mean()
    loss.backward()


def test_constructor_FourierFeatureEmbedding():
    FourierFeatureEmbedding(input_dimension=1, output_dimension=20, sigma=1)
    with pytest.raises(TypeError):
        FourierFeatureEmbedding()
    with pytest.raises(RuntimeError):
        FourierFeatureEmbedding(input_dimension=1, output_dimension=3, sigma=1)
    with pytest.raises(ValueError):
        FourierFeatureEmbedding(
            input_dimension="x", output_dimension=20, sigma=1
        )
        FourierFeatureEmbedding(
            input_dimension=1, output_dimension="x", sigma=1
        )
        FourierFeatureEmbedding(
            input_dimension=1, output_dimension=20, sigma="x"
        )


@pytest.mark.parametrize("output_dimension", [2, 4, 6])
@pytest.mark.parametrize("input_dimension", [1, 2, 3])
@pytest.mark.parametrize("sigma", [10, 1, 0.1])
def test_forward_backward_FourierFeatureEmbedding(
    input_dimension, output_dimension, sigma
):
    func = FourierFeatureEmbedding(input_dimension, output_dimension, sigma)
    # coordinates
    x = torch.rand((10, input_dimension), requires_grad=True)
    # output
    f = func(x)
    assert f.shape[-1] == output_dimension
    # compute backward
    loss = f.mean()
    loss.backward()
