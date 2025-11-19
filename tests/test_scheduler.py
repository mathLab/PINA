import torch
import pytest
from pina.optim import TorchOptimizer, TorchScheduler

model = torch.nn.Linear(10, 10)
opt_list = [
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.SGD,
    torch.optim.RMSprop,
]

sch_list = [torch.optim.lr_scheduler.ConstantLR]


@pytest.mark.parametrize("scheduler_class", sch_list)
def test_constructor(scheduler_class):
    TorchScheduler(scheduler_class)


@pytest.mark.parametrize("optimizer_class", opt_list)
@pytest.mark.parametrize("scheduler_class", sch_list)
def test_hook(optimizer_class, scheduler_class):
    opt = TorchOptimizer(optimizer_class, lr=1e-3)
    opt.parameter_hook(model.parameters())
    sch = TorchScheduler(scheduler_class)
    assert sch.hooks_done["optimizer_hook"] is False
    sch.optimizer_hook(opt)
    assert sch.hooks_done["optimizer_hook"] is True


@pytest.mark.parametrize("optimizer_class", opt_list)
@pytest.mark.parametrize("scheduler_class", sch_list)
def test_instance(optimizer_class, scheduler_class):
    opt = TorchOptimizer(optimizer_class, lr=1e-3)
    opt.parameter_hook(model.parameters())
    sch = TorchScheduler(scheduler_class)
    sch.optimizer_hook(opt)
    assert isinstance(sch.instance, scheduler_class)
