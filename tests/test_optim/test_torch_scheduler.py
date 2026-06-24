import torch
import pytest
from pina.optim import TorchOptimizer, TorchScheduler

opt_list = [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD]
sch_list = [
    torch.optim.lr_scheduler.ConstantLR,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
]


@pytest.mark.parametrize("scheduler_class", sch_list)
def test_constructor(scheduler_class):
    TorchScheduler(scheduler_class)

    # Should fail if the scheduler is not subclass of torch LRScheduler
    with pytest.raises(ValueError):
        TorchScheduler(object)


@pytest.mark.parametrize("optimizer_class", opt_list)
@pytest.mark.parametrize("scheduler_class", sch_list)
def test_hook(optimizer_class, scheduler_class):

    # Create the optimizer instance
    optimizer = TorchOptimizer(optimizer_class)
    optimizer.hook(torch.nn.Linear(10, 10).parameters())

    # Create the scheduler instance
    scheduler = TorchScheduler(scheduler_class)

    # Hook the scheduler with the optimizer instance
    scheduler.hook(optimizer)

    # Should fail if the optimizer is not an instance of OptimizerInterface
    with pytest.raises(ValueError):
        scheduler.hook(object)
