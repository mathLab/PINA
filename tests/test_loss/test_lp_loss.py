import torch
import pytest
from pina.loss import LpLoss

# Define input and target for tests
input = torch.rand(10, 2)
target = torch.rand(10, 2)


@pytest.mark.parametrize("p", [2, 0.5, "inf", "-inf"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("relative", [True, False])
def test_constructor(p, reduction, relative):

    # Define the loss
    LpLoss(p=p, reduction=reduction, relative=relative)

    # Should fail if p is invalid
    with pytest.raises(ValueError):
        LpLoss(p="invalid", reduction=reduction, relative=relative)

    # Should fail if reduction is invalid
    with pytest.raises(ValueError):
        LpLoss(p=p, reduction="invalid", relative=relative)

    # Should fail if relative is not a boolean
    with pytest.raises(ValueError):
        LpLoss(p=p, reduction=reduction, relative="invalid")


@pytest.mark.parametrize("p", [2, 0.5, "inf", "-inf"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("relative", [True, False])
def test_forward(p, reduction, relative):

    # Define the loss
    loss = LpLoss(p=p, reduction=reduction, relative=relative)

    # Forward pass
    value = loss(input, target)

    # Check shape
    if loss.reduction != "none":
        assert value.shape == torch.Size([1])
    else:
        assert value.shape == torch.Size([target.shape[0]])
