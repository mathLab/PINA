
import torch
import pytest
from pina import TorchOptimizer, TorchScheduler

opt_list = [
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.SGD,
    torch.optim.RMSprop
]

sch_list = [
    torch.optim.lr_scheduler.ConstantLR
]

@pytest.mark.parametrize("scheduler_class", sch_list)
def test_constructor(scheduler_class):
    TorchScheduler(scheduler_class)

@pytest.mark.parametrize("optimizer_class", opt_list)
@pytest.mark.parametrize("scheduler_class", sch_list)
def test_hook(optimizer_class, scheduler_class):
    opt = TorchOptimizer(optimizer_class, lr=1e-3)
    opt.hook(torch.nn.Linear(10, 10).parameters())
    sch = TorchScheduler(scheduler_class)
    sch.hook(opt)