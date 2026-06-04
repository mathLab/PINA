import torch
import pytest
from pina.solver import PhysicsInformedSingleModelSolver
from pina.model import FeedForward
from pina.callback import MetricTracker
from pina import Trainer, Condition, LabelTensor
from pina.problem.zoo import Poisson2DSquareProblem

# Initialize the problem
problem = Poisson2DSquareProblem()
problem.discretise_domain(10, "random")
problem.conditions["data"] = Condition(
    input=LabelTensor(torch.randn(10, 2), labels=["x", "y"]),
    target=LabelTensor(torch.randn(10, 1), labels=["u"]),
)

# Initialize the model and solver
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
solver = PhysicsInformedSingleModelSolver(problem=problem, model=model)


@pytest.mark.parametrize(
    "metrics_to_track", [["D_loss", "train_loss"], "data_loss", None]
)
def test_constructor(metrics_to_track):
    MetricTracker(metrics_to_track=metrics_to_track)

    # Should fail if metrics_to_track is not a string or list of strings
    with pytest.raises(ValueError):
        MetricTracker(metrics_to_track=123)


@pytest.mark.parametrize(
    "metrics_to_track", [["D_loss", "train_loss"], "data_loss", None]
)
@pytest.mark.parametrize("batch_size", [None, 8])
def test_routine(metrics_to_track, batch_size):

    # Initialize the callback
    callback = MetricTracker(metrics_to_track=metrics_to_track)

    # Convert to list if a single string is provided
    if isinstance(metrics_to_track, str):
        metrics_to_track = [metrics_to_track]

    # Initialize the trainer with the callback and train the model
    trainer = Trainer(
        solver=solver,
        callbacks=[callback],
        accelerator="cpu",
        max_epochs=5,
        batch_size=batch_size,
        log_every_n_steps=1,
    )
    trainer.train()

    # Get the logged metrics from the callback
    logged_metrics = sorted(list(trainer.callbacks[0].metrics.keys()))

    # Define the expected metrics
    expected_metrics = metrics_to_track or ["train_loss"]

    # If a batch size is provided, expand metric names to match convention
    if batch_size is not None:
        expected_metrics = [
            f"{metric}_{suffix}"
            for metric in expected_metrics
            for suffix in ("step", "epoch")
        ]

    assert sorted(logged_metrics) == sorted(expected_metrics)
