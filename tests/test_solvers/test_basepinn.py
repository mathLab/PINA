import torch
import pytest

from pina import Trainer
from pina.optim import TorchOptimizer, TorchScheduler
from pina.model import FeedForward
from pina.solvers import PINNInterface
from pina.problem import InverseProblem
from pina.problem.zoo.poisson_2d_square import (
    Poisson2DSquareProblem as Poisson
)

# Define a basic solver implementing PINNInterface
class FooPINN(PINNInterface):
    def __init__(self, problem, model):
        super().__init__(problem=problem)
        self.models = [model]
        self.optimizer = TorchOptimizer(torch.optim.Adam, lr=0.001)
        self.scheduler = TorchScheduler(torch.optim.lr_scheduler.ConstantLR)

    def forward(self, x):
        return self.models[0](x)

    def loss_phys(self, samples, equation):
        residual = self.compute_residual(samples=samples, equation=equation)
        loss_value = self.loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        return loss_value

    # Needed for testing
    def configure_optimizers(self):
        if isinstance(self.problem, InverseProblem):
            self.optimizer.optimizer_instance.add_param_group(
                    {
                        "params": [
                            self._params[var]
                            for var in self.problem.unknown_variables
                        ]
                    }
                )
        self.optimizer.hook(self.models[0].parameters())
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.optimizer_instance],
                [self.scheduler.scheduler_instance])


# Define the problem
poisson_problem = Poisson()
poisson_problem.discretise_domain(100)

# Define the model
model = FeedForward(
    len(poisson_problem.input_variables),
    len(poisson_problem.output_variables)
)

def test_constructor():
    with pytest.raises(TypeError):
        PINNInterface()
    FooPINN(poisson_problem, model)


def test_train_step():
    solver = FooPINN(poisson_problem, model)
    trainer = Trainer(solver, max_epochs=2, accelerator='cpu')
    trainer.train()
    trainer.test()


test_train_step()