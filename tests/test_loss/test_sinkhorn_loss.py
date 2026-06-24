import torch
import pytest
from pina.loss import SinkhornLoss


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("eps", [0.01, 1])
@pytest.mark.parametrize("iterations", [2, 5])
def test_constructor(p, eps, iterations):

    # Define the loss
    SinkhornLoss(p=p, eps=eps, iterations=iterations)

    # Should fail if iterations is not a positive integer
    with pytest.raises(AssertionError):
        SinkhornLoss(p=p, eps=eps, iterations=0)

    # Should fail if p is not a positive integer
    with pytest.raises(AssertionError):
        SinkhornLoss(p=0, eps=eps, iterations=iterations)

    # Should fail if eps is not numeric
    with pytest.raises(ValueError):
        SinkhornLoss(p=p, eps="invalid", iterations=iterations)

    # Should fail if eps is not positive
    with pytest.raises(ValueError):
        SinkhornLoss(p=p, eps=-0.1, iterations=iterations)


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("eps", [0.1, 1])
@pytest.mark.parametrize("iterations", [2, 5])
@pytest.mark.parametrize(
    "input, target",
    [
        (torch.rand(10, 2), torch.rand(8, 2)),
        (torch.rand(5, 3), torch.rand(5, 3)),
        (torch.rand(1, 4), torch.rand(7, 4)),
        (torch.rand(6, 4), torch.rand(1, 4)),
        (torch.rand(3, 1), torch.rand(4, 1)),
    ],
)
def test_forward(p, eps, iterations, input, target):

    # Define the loss
    loss = SinkhornLoss(p=p, eps=eps, iterations=iterations)

    # Forward pass
    value = loss(input, target)

    # Check shape
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
