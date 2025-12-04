import torch
import pytest

from pina.solver import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.optim import TorchScheduler
from pina.callback import SwitchScheduler
from pina.problem.zoo import Poisson2DSquareProblem as Poisson


# Define the problem
problem = Poisson()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))

# Define the scheduler
scheduler = TorchScheduler(torch.optim.lr_scheduler.ConstantLR, factor=0.1)

# Initialize the solver
solver = PINN(problem=problem, model=model, scheduler=scheduler)

# Define new schedulers for testing
step = TorchScheduler(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.1)
exp = TorchScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9)


@pytest.mark.parametrize("epoch_switch", [5, 10])
@pytest.mark.parametrize("new_sched", [step, exp])
def test_switch_scheduler_constructor(new_sched, epoch_switch):

    # Constructor
    SwitchScheduler(new_schedulers=new_sched, epoch_switch=epoch_switch)

    # Should fail if epoch_switch is less than 1
    with pytest.raises(AssertionError):
        SwitchScheduler(new_schedulers=new_sched, epoch_switch=0)


@pytest.mark.parametrize("epoch_switch", [5, 10])
@pytest.mark.parametrize("new_sched", [step, exp])
def test_switch_scheduler_routine(new_sched, epoch_switch):

    # Initialize the trainer
    switch_sched_callback = SwitchScheduler(
        new_schedulers=new_sched, epoch_switch=epoch_switch
    )
    trainer = Trainer(
        solver=solver,
        callbacks=switch_sched_callback,
        accelerator="cpu",
        max_epochs=epoch_switch + 2,
    )
    trainer.train()

    # Check that the solver and trainer strategy schedulers have been updated
    assert solver.scheduler.instance.__class__ == new_sched.instance.__class__
    assert (
        trainer.lr_scheduler_configs[0].scheduler.__class__
        == new_sched.instance.__class__
    )
