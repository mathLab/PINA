import pytest
import torch
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.loss import ScalarWeighting
from pina.problem.zoo import Poisson2DSquareProblem


# Initialize problem and model
problem = Poisson2DSquareProblem()
problem.discretise_domain(50)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
condition_names = problem.conditions.keys()


@pytest.mark.parametrize(
    "weights", [1, 1.0, dict(zip(condition_names, [1] * len(condition_names)))]
)
def test_constructor(weights):
    ScalarWeighting(weights=weights)

    # Should fail if weights are not a scalar
    with pytest.raises(ValueError):
        ScalarWeighting(weights="invalid")

    # Should fail if weights are not a dictionary
    with pytest.raises(ValueError):
        ScalarWeighting(weights=[1, 2, 3])


@pytest.mark.parametrize(
    "weights", [1, 1.0, dict(zip(condition_names, [1] * len(condition_names)))]
)
def test_aggregate(weights):
    weighting = ScalarWeighting(weights=weights)
    losses = dict(
        zip(
            condition_names,
            [torch.randn(1) for _ in range(len(condition_names))],
        )
    )
    weighting.aggregate(losses=losses)


@pytest.mark.parametrize(
    "weights", [1, 1.0, dict(zip(condition_names, [1] * len(condition_names)))]
)
def test_train_aggregation(weights):
    weighting = ScalarWeighting(weights=weights)
    solver = PINN(problem=problem, model=model, weighting=weighting)
    trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    trainer.train()
