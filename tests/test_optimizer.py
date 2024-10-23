import torch
import pytest
from pina import TorchOptimizer

opt_list = [
    torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.RMSprop
]


@pytest.mark.parametrize("optimizer_class", opt_list)
def test_constructor(optimizer_class):
    TorchOptimizer(optimizer_class, lr=1e-3)


@pytest.mark.parametrize("optimizer_class", opt_list)
def test_hook(optimizer_class):
    opt = TorchOptimizer(optimizer_class, lr=1e-3)
    opt.hook(torch.nn.Linear(10, 10).parameters())
