import torch
import pytest

from pina.solver import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.optim import TorchOptimizer
from pina.callback import SwitchOptimizer
from pina.problem.zoo import Poisson2DSquareProblem as Poisson


# Define the problem
problem = Poisson()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))

# Define the optimizer
optimizer = TorchOptimizer(torch.optim.Adam)

# Initialize the solver
solver = PINN(problem=problem, model=model, optimizer=optimizer)

# Define new optimizers for testing
lbfgs = TorchOptimizer(torch.optim.LBFGS, lr=1.0)
adamW = TorchOptimizer(torch.optim.AdamW, lr=0.01)


@pytest.mark.parametrize("epoch_switch", [5, 10])
@pytest.mark.parametrize("new_opt", [lbfgs, adamW])
def test_switch_optimizer_constructor(new_opt, epoch_switch):

    # Constructor
    SwitchOptimizer(new_optimizers=new_opt, epoch_switch=epoch_switch)

    # Should fail if epoch_switch is less than 1
    with pytest.raises(ValueError):
        SwitchOptimizer(new_optimizers=new_opt, epoch_switch=0)


@pytest.mark.parametrize("epoch_switch", [5, 10])
@pytest.mark.parametrize("new_opt", [lbfgs, adamW])
def test_switch_optimizer_routine(new_opt, epoch_switch):

    # Check if the optimizer is initialized correctly
    solver.configure_optimizers()

    # Initialize the trainer
    switch_opt_callback = SwitchOptimizer(
        new_optimizers=new_opt, epoch_switch=epoch_switch
    )
    trainer = Trainer(
        solver=solver,
        callbacks=switch_opt_callback,
        accelerator="cpu",
        max_epochs=epoch_switch + 2,
    )
    trainer.train()

    # Check that the trainer strategy optimizers have been updated
    assert solver.optimizer.instance.__class__ == new_opt.instance.__class__
    assert (
        trainer.strategy.optimizers[0].__class__ == new_opt.instance.__class__
    )
