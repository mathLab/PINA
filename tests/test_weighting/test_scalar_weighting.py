import torch
import pytest
from pina import Trainer
from pina.solver import PhysicsInformedSingleModelSolver
from pina.model import FeedForward
from pina.weighting import ScalarWeighting
from pina.problem.zoo import Poisson2DSquareProblem

# Initialize problem and model
problem = Poisson2DSquareProblem()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
condition_names = problem.conditions.keys()

# Weight dictionaries for testing
weights_dict_1 = dict(zip(condition_names, [1] * len(condition_names)))
weights_dict_2 = {cond: torch.rand(1).item() * 10 for cond in condition_names}


@pytest.mark.parametrize("weights", [1, 3.0, weights_dict_1, weights_dict_2])
def test_constructor(weights):
    ScalarWeighting(weights=weights)

    # Should fail if weights are not a scalar
    with pytest.raises(ValueError):
        ScalarWeighting(weights="invalid")

    # Should fail if weights are not a dictionary
    with pytest.raises(ValueError):
        ScalarWeighting(weights=[1, 2, 3])


@pytest.mark.parametrize("weights", [1, 3.0, weights_dict_1, weights_dict_2])
def test_aggregate(weights):

    # Initialize weighting, solver, and trainer
    weighting = ScalarWeighting(weights=weights)
    solver = PhysicsInformedSingleModelSolver(
        problem=problem, model=model, weighting=weighting
    )
    trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")

    # Train
    trainer.train()

    # Check that weights keys are the same as loss keys
    assert weighting.last_saved_weights().keys() == problem.conditions.keys()

    # Check that weights values are correct
    for condition in problem.conditions.keys():
        expected_weight = (
            weights[condition] if isinstance(weights, dict) else weights
        )
        assert weighting.last_saved_weights()[condition] == expected_weight
