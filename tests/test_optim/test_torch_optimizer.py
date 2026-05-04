import torch
import pytest
from pina.optim import TorchOptimizer

opt_list = [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD]
kwargs_list = [{"lr": 1e-3}, {"lr": 1e-3, "weight_decay": 1e-4}]


@pytest.mark.parametrize("optimizer_class", opt_list)
@pytest.mark.parametrize("kwargs", kwargs_list)
def test_constructor(optimizer_class, kwargs):
    TorchOptimizer(optimizer_class, **kwargs)

    # Should fail if the optimizer is not subclass of torch.optim.Optimizer
    with pytest.raises(ValueError):
        TorchOptimizer(object, **kwargs)


@pytest.mark.parametrize("optimizer_class", opt_list)
@pytest.mark.parametrize("kwargs", kwargs_list)
def test_hook(optimizer_class, kwargs):

    # Create the optimizer instance
    optimizer = TorchOptimizer(optimizer_class, **kwargs)

    # Hook the optimizer with model parameters
    optimizer.hook(torch.nn.Linear(10, 10).parameters())
