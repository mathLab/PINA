import torch
import pytest
from pina.solver import SupervisedSolver
from pina.problem.zoo import SupervisedProblem
from pina.optim import TorchOptimizer


problem = SupervisedProblem(torch.randn(2, 10), torch.randn(2, 10))
model = torch.nn.Linear(10, 10)
opt_list = [
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.SGD,
    torch.optim.RMSprop,
]


@pytest.mark.parametrize("optimizer_class", opt_list)
def test_constructor(optimizer_class):
    TorchOptimizer(optimizer_class, lr=1e-3)


@pytest.mark.parametrize("optimizer_class", opt_list)
def test_parameter_hook(optimizer_class):
    opt = TorchOptimizer(optimizer_class, lr=1e-3)
    assert opt.hooks_done["parameter_hook"] is False
    opt.parameter_hook(model.parameters())
    assert opt.hooks_done["parameter_hook"] is True


@pytest.mark.parametrize("optimizer_class", opt_list)
def test_solver_hook(optimizer_class):
    opt = TorchOptimizer(optimizer_class, lr=1e-3)
    solver = SupervisedSolver(problem=problem, model=model, optimizer=opt)
    assert opt.hooks_done["solver_hook"] is False
    with pytest.raises(RuntimeError):
        opt.solver_hook(solver)
    solver.configure_optimizers()
    assert opt.hooks_done["solver_hook"] is True
    assert opt.solver is solver


@pytest.mark.parametrize("optimizer_class", opt_list)
def test_instance(optimizer_class):
    opt = TorchOptimizer(optimizer_class, lr=1e-3)
    opt.parameter_hook(model.parameters())
    assert isinstance(opt.instance, optimizer_class)
