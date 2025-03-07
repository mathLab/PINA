import torch
import pytest

from pina.loss import PowerLoss

input = torch.tensor([[3.0], [1.0], [-8.0]])
target = torch.tensor([[6.0], [4.0], [2.0]])
available_reductions = ["str", "mean", "none"]


def test_PowerLoss_constructor():
    # test reduction
    for reduction in available_reductions:
        PowerLoss(reduction=reduction)
    # test p
    for p in [float("inf"), -float("inf"), 1, 10, -8]:
        PowerLoss(p=p)


def test_PowerLoss_forward():
    # l2 loss
    loss = PowerLoss(p=2, reduction="mean")
    l2_loss = torch.mean((input - target).pow(2))
    assert loss(input, target) == l2_loss
    # l1 loss
    loss = PowerLoss(p=1, reduction="sum")
    l1_loss = torch.sum(torch.abs(input - target))
    assert loss(input, target) == l1_loss


def test_LpRelativeLoss_constructor():
    # test reduction
    for reduction in available_reductions:
        PowerLoss(reduction=reduction, relative=True)
    # test p
    for p in [float("inf"), -float("inf"), 1, 10, -8]:
        PowerLoss(p=p, relative=True)


def test_LpRelativeLoss_forward():
    # l2 relative loss
    loss = PowerLoss(p=2, reduction="mean", relative=True)
    l2_loss = (input - target).pow(2) / input.pow(2)
    assert loss(input, target) == torch.mean(l2_loss)
    # l1 relative loss
    loss = PowerLoss(p=1, reduction="sum", relative=True)
    l1_loss = torch.abs(input - target) / torch.abs(input)
    assert loss(input, target) == torch.sum(l1_loss)
