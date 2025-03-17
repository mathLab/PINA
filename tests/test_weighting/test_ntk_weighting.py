import pytest
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem
from pina.loss import Neural_Tangent_Kernel

problem = Poisson2DSquareProblem()
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
condition_names = problem.conditions.keys()
#print(problem.conditions.keys())

def test_constructor(model,alpha):
    Neural_Tangent_Kernel(model=model,alpha=alpha)

def test_wrong_constructor1(alpha):
    with pytest.raises(ValueError):
        Neural_Tangent_Kernel(model,alpha)

def test_wrong_constructor2(model):
    with pytest.raises(TypeError):
        Neural_Tangent_Kernel(model)

def test_train_aggregation(model):
    weighting = Neural_Tangent_Kernel(model=model,alpha = 0.5)
    problem.discretise_domain(50)
    solver = PINN(problem=problem, model=model, weighting=weighting)
    trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    trainer.train()

