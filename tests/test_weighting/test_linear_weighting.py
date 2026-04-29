import math
import torch
import pytest
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.weighting import LinearWeighting
from pina.problem.zoo import Poisson2DSquareProblem

# Initialize problem and model
problem = Poisson2DSquareProblem()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
condition_names = problem.conditions.keys()

# Weight dictionaries for testing
init_dict_1 = {cond: torch.rand(1).item() * 10 for cond in condition_names}
init_dict_2 = {cond: torch.rand(1).item() * 10 for cond in condition_names}
final_dict_1 = {cond: torch.rand(1).item() * 1 for cond in condition_names}
final_dict_2 = {cond: torch.rand(1).item() * 100 for cond in condition_names}


@pytest.mark.parametrize("initial_weights", [init_dict_1, init_dict_2])
@pytest.mark.parametrize("final_weights", [final_dict_1, final_dict_2])
@pytest.mark.parametrize("target_epoch", [5, 10])
def test_constructor(initial_weights, final_weights, target_epoch):
    LinearWeighting(
        initial_weights=initial_weights,
        final_weights=final_weights,
        target_epoch=target_epoch,
    )

    # Should fail if initial_weights is not a dictionary
    with pytest.raises(ValueError):
        LinearWeighting(
            initial_weights=[1, 1, 1],
            final_weights=final_weights,
            target_epoch=target_epoch,
        )

    # Should fail if final_weights is not a dictionary
    with pytest.raises(ValueError):
        LinearWeighting(
            initial_weights=initial_weights,
            final_weights=[1, 1, 1],
            target_epoch=target_epoch,
        )

    # Should fail if target_epoch is not an integer
    with pytest.raises(AssertionError):
        LinearWeighting(
            initial_weights=initial_weights,
            final_weights=final_weights,
            target_epoch=1.5,
        )

    # Should fail if target_epoch is not positive
    with pytest.raises(AssertionError):
        LinearWeighting(
            initial_weights=initial_weights,
            final_weights=final_weights,
            target_epoch=0,
        )

    # Should fail if dictionary keys do not match
    with pytest.raises(ValueError):
        LinearWeighting(
            initial_weights={"invalid": 1},
            final_weights=final_weights,
            target_epoch=target_epoch,
        )


@pytest.mark.parametrize("initial_weights", [init_dict_1, init_dict_2])
@pytest.mark.parametrize("final_weights", [final_dict_1, final_dict_2])
@pytest.mark.parametrize("target_epoch", [5, 10])
def test_train_aggregation(initial_weights, final_weights, target_epoch):

    # Initialize weighting, solver, and trainer
    weighting = LinearWeighting(
        initial_weights=initial_weights,
        final_weights=final_weights,
        target_epoch=target_epoch,
    )
    solver = PINN(problem=problem, model=model, weighting=weighting)
    trainer = Trainer(
        solver=solver,
        max_epochs=target_epoch + torch.randint(1, 5, (1,)).item(),
        accelerator="cpu",
    )

    # Train
    trainer.train()

    # Check that weights keys are the same as loss keys
    assert weighting.last_saved_weights().keys() == problem.conditions.keys()

    # Check that the weights have been updated correctly at each epoch
    epoch = min(solver.trainer.current_epoch, target_epoch)
    progress = epoch / target_epoch

    # Check that the weights are updated according to linear interpolation
    for condition in problem.conditions.keys():

        # Initial and final weights for the condition
        initial_weight = initial_weights[condition]
        final_weight = final_weights[condition]

        # Compute the expected weight based on linear interpolation
        expected_weight = (
            initial_weight + (final_weight - initial_weight) * progress
        )

        assert math.isclose(
            weighting.last_saved_weights()[condition],
            expected_weight,
            rel_tol=1e-5,
        )
