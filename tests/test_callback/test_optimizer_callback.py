from pina.callback import SwitchOptimizer
import torch
import pytest

from pina.solver import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from pina.optim import TorchOptimizer

# make the problem
poisson_problem = Poisson()
boundaries = ['g1', 'g2', 'g3', 'g4']
n = 10
poisson_problem.discretise_domain(n, 'grid', domains=boundaries)
poisson_problem.discretise_domain(n, 'grid', domains='D')
model = FeedForward(len(poisson_problem.input_variables),
                    len(poisson_problem.output_variables))

# make the solver
solver = PINN(problem=poisson_problem, model=model)

adam_optimizer = TorchOptimizer(torch.optim.Adam, lr=0.01)
lbfgs_optimizer = TorchOptimizer(torch.optim.LBFGS, lr= 0.001)

def test_switch_optimizer_constructor():
    SwitchOptimizer(adam_optimizer, epoch_switch=10)


# def test_switch_optimizer_routine(): #TODO revert
#     # make the trainer
#     switch_opt_callback = SwitchOptimizer(lbfgs_optimizer, epoch_switch=3)
#     trainer = Trainer(solver=solver,
#                       callback=[switch_opt_callback],
#                       accelerator='cpu',
#                       max_epochs=5)
#     trainer.train()
