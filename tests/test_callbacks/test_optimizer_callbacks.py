from pina.callbacks import SwitchOptimizer
import torch
import pytest

from pina.solvers import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem as Poisson

# make the problem
poisson_problem = Poisson()
boundaries = ['nil_g1', 'nil_g2', 'nil_g3', 'nil_g4']
n = 10
poisson_problem.discretise_domain(n, 'grid', locations=boundaries)
model = FeedForward(len(poisson_problem.input_variables),
                    len(poisson_problem.output_variables))

# make the solver
solver = PINN(problem=poisson_problem, model=model)


def test_switch_optimizer_constructor():
    SwitchOptimizer(new_optimizers=torch.optim.Adam,
                    new_optimizers_kwargs={'lr': 0.01},
                    epoch_switch=10)

    with pytest.raises(ValueError):
        SwitchOptimizer(new_optimizers=[torch.optim.Adam, torch.optim.Adam],
                        new_optimizers_kwargs=[{
                            'lr': 0.01
                        }],
                        epoch_switch=10)


def test_switch_optimizer_routine():
    # make the trainer
    trainer = Trainer(solver=solver,
                      callbacks=[
                          SwitchOptimizer(new_optimizers=torch.optim.LBFGS,
                                          new_optimizers_kwargs={'lr': 0.01},
                                          epoch_switch=3)
                      ],
                      accelerator='cpu',
                      max_epochs=5)
    trainer.train()
