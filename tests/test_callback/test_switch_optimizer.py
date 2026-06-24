import torch
import pytest
from pina.solver import PhysicsInformedSingleModelSolver
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.optim import TorchOptimizer
from pina.callback import SwitchOptimizer
from pina.problem.zoo import Poisson2DSquareProblem

# Define the problem
problem = Poisson2DSquareProblem()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))

# Define the optimizer
optimizer = TorchOptimizer(torch.optim.Adam)

# Initialize the solver
solver = PhysicsInformedSingleModelSolver(
    problem=problem, model=model, optimizer=optimizer
)

# Define new optimizers for testing
lbfgs = TorchOptimizer(torch.optim.LBFGS, lr=1.0)
adamW = TorchOptimizer(torch.optim.AdamW, lr=0.01)


@pytest.mark.parametrize("epoch_switch", [5, 10])
@pytest.mark.parametrize("new_opt", [lbfgs, adamW])
def test_constructor(new_opt, epoch_switch):

    # Constructor
    SwitchOptimizer(new_optimizers=new_opt, epoch_switch=epoch_switch)

    # Should fail if epoch_switch is not a positive integer
    with pytest.raises(AssertionError):
        SwitchOptimizer(new_optimizers=new_opt, epoch_switch=0)

    # Should fail if new_optimizers is not an instance of OptimizerInterface
    with pytest.raises(ValueError):
        SwitchOptimizer(
            new_optimizers="not_an_optimizer", epoch_switch=epoch_switch
        )


@pytest.mark.parametrize("epoch_switch", [5, 10])
@pytest.mark.parametrize("new_opt", [lbfgs, adamW])
def test_routine(new_opt, epoch_switch):

    # Check if the optimizer is initialized correctly
    solver.configure_optimizers()

    # Initialize the callback
    switch_opt_callback = SwitchOptimizer(
        new_optimizers=new_opt, epoch_switch=epoch_switch
    )

    # Initialize the trainer
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
