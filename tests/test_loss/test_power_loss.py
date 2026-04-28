import torch
import pytest
from pina.loss import PowerLoss

# Define input and target for tests
input = torch.rand(10, 2)
target = torch.rand(10, 2)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("relative", [True, False])
def test_constructor(p, reduction, relative):

    # Define the loss
    PowerLoss(p=p, reduction=reduction, relative=relative)

    # Should fail if p is not a positive integer
    with pytest.raises(AssertionError):
        PowerLoss(p=-2, reduction=reduction, relative=relative)

    # Should fail if reduction is invalid
    with pytest.raises(ValueError):
        PowerLoss(p=p, reduction="invalid", relative=relative)

    # Should fail if relative is not a boolean
    with pytest.raises(ValueError):
        PowerLoss(p=p, reduction=reduction, relative="invalid")


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("relative", [True, False])
def test_forward(p, reduction, relative):

    # Define the loss
    loss = PowerLoss(p=p, reduction=reduction, relative=relative)

    # Forward pass
    value = loss(input, target)

    # Check shape
    if loss.reduction != "none":
        assert value.shape == torch.Size([1])
    else:
        assert value.shape == torch.Size([target.shape[0]])
