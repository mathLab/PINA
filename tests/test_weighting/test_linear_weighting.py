import math
import pytest
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.loss import LinearWeighting
from pina.problem.zoo import Poisson2DSquareProblem


# Initialize problem and model
problem = Poisson2DSquareProblem()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))

# Weights for testing
init_weight_1 = {cond: 3 for cond in problem.conditions.keys()}
init_weight_2 = {cond: 4 for cond in problem.conditions.keys()}
final_weight_1 = {cond: 1 for cond in problem.conditions.keys()}
final_weight_2 = {cond: 5 for cond in problem.conditions.keys()}


@pytest.mark.parametrize("initial_weights", [init_weight_1, init_weight_2])
@pytest.mark.parametrize("final_weights", [final_weight_1, final_weight_2])
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
            initial_weights={list(initial_weights.keys())[0]: 1},
            final_weights=final_weights,
            target_epoch=target_epoch,
        )


@pytest.mark.parametrize("initial_weights", [init_weight_1, init_weight_2])
@pytest.mark.parametrize("final_weights", [final_weight_1, final_weight_2])
@pytest.mark.parametrize("target_epoch", [5, 10])
def test_train_aggregation(initial_weights, final_weights, target_epoch):
    weighting = LinearWeighting(
        initial_weights=initial_weights,
        final_weights=final_weights,
        target_epoch=target_epoch,
    )
    solver = PINN(problem=problem, model=model, weighting=weighting)
    trainer = Trainer(solver=solver, max_epochs=target_epoch, accelerator="cpu")
    trainer.train()

    # Check that weights are updated correctly
    assert all(
        math.isclose(
            weighting.last_saved_weights()[cond],
            final_weights[cond],
            rel_tol=1e-5,
            abs_tol=1e-8,
        )
        for cond in final_weights.keys()
    )
