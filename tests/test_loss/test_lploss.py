import torch

from pina.loss import LpLoss

input = torch.tensor([[3.], [1.], [-8.]])
target = torch.tensor([[6.], [4.], [2.]])
available_reductions = ['str', 'mean', 'none']


def test_LpLoss_constructor():
    # test reduction
    for reduction in available_reductions:
        LpLoss(reduction=reduction)
    # test p
    for p in [float('inf'), -float('inf'), 1, 10, -8]:
        LpLoss(p=p)


def test_LpLoss_forward():
    # l2 loss
    loss = LpLoss(p=2, reduction='mean')
    l2_loss = torch.mean(torch.sqrt((input - target).pow(2)))
    assert loss(input, target) == l2_loss
    # l1 loss
    loss = LpLoss(p=1, reduction='sum')
    l1_loss = torch.sum(torch.abs(input - target))
    assert loss(input, target) == l1_loss


def test_LpRelativeLoss_constructor():
    # test reduction
    for reduction in available_reductions:
        LpLoss(reduction=reduction, relative=True)
    # test p
    for p in [float('inf'), -float('inf'), 1, 10, -8]:
        LpLoss(p=p, relative=True)


def test_LpRelativeLoss_forward():
    # l2 relative loss
    loss = LpLoss(p=2, reduction='mean', relative=True)
    l2_loss = torch.sqrt((input - target).pow(2)) / torch.sqrt(input.pow(2))
    assert loss(input, target) == torch.mean(l2_loss)
    # l1 relative loss
    loss = LpLoss(p=1, reduction='sum', relative=True)
    l1_loss = torch.abs(input - target) / torch.abs(input)
    assert loss(input, target) == torch.sum(l1_loss)
