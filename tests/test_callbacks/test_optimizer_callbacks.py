from pina.callbacks import SwitchOptimizer
import torch
import pytest

from pina.solvers import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from pina.optim import TorchOptimizer

# make the problem
poisson_problem = Poisson()
boundaries = ['nil_g1', 'nil_g2', 'nil_g3', 'nil_g4']
n = 10
poisson_problem.discretise_domain(n, 'grid', locations=boundaries)
poisson_problem.discretise_domain(n, 'grid', locations='laplace_D')
model = FeedForward(len(poisson_problem.input_variables),
                    len(poisson_problem.output_variables))

# make the solver
solver = PINN(problem=poisson_problem, model=model)

adam_optimizer = TorchOptimizer(torch.optim.Adam, lr=0.01)
lbfgs_optimizer = TorchOptimizer(torch.optim.LBFGS, lr= 0.001)

def test_switch_optimizer_constructor():
    SwitchOptimizer(adam_optimizer, epoch_switch=10)


def test_switch_optimizer_routine():
    # make the trainer
    switch_opt_callback = SwitchOptimizer(lbfgs_optimizer, epoch_switch=3)
    trainer = Trainer(solver=solver,
                      callbacks=[switch_opt_callback],
                      accelerator='cpu',
                      max_epochs=5)
    trainer.train()
