import pytest
import torch

from pina import Trainer
from pina.solvers import PINN
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem
from pina.loss import ScalarWeighting

problem = Poisson2DSquareProblem()
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
condition_names = problem.conditions.keys()
print(problem.conditions.keys())

@pytest.mark.parametrize("weights",
                         [1, 1., dict(zip(condition_names, [1]*len(condition_names)))])
def test_constructor(weights):
    ScalarWeighting(weights=weights)

@pytest.mark.parametrize("weights", ['a', [1,2,3]])
def test_wrong_constructor(weights):
    with pytest.raises(ValueError):
        ScalarWeighting(weights=weights)

@pytest.mark.parametrize("weights",
                         [1, 1., dict(zip(condition_names, [1]*len(condition_names)))])
def test_aggregate(weights):
    weighting = ScalarWeighting(weights=weights)
    losses = dict(zip(condition_names, [torch.randn(1) for _ in range(len(condition_names))]))
    weighting.aggregate(losses=losses)

@pytest.mark.parametrize("weights",
                         [1, 1., dict(zip(condition_names, [1]*len(condition_names)))])
def test_train_aggregation(weights):
    weighting = ScalarWeighting(weights=weights)
    problem.discretise_domain(50)
    solver = PINN(
                problem=problem,
                model=model,
                weighting=weighting)
    trainer = Trainer(solver=solver, max_epochs=5, accelerator='cpu')
    trainer.train()