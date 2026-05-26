import torch
import pytest
from pina.solver import PhysicsInformedSingleModelSolver
from pina.model import FeedForward
from pina.callback import PINAProgressBar
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

# Define metrics to be used in the progress bar
metrics_list = ["train", "val", "test", ["test", "data"], ["train", "val"]]


@pytest.mark.parametrize(
    "metrics", ["train", "val", "test", ["test", "data"], ["train", "val"]]
)
def test_constructor(metrics):
    PINAProgressBar(metrics=metrics)

    # Should fail if metrics is not a string or list of strings
    with pytest.raises(ValueError):
        PINAProgressBar(metrics=123)


@pytest.mark.parametrize(
    "metrics",
    [
        "train",
        "val",
        "test",
        ["test", "data"],
        ["train", "val"],
    ],
)
@pytest.mark.parametrize("batch_size", [None, 8])
def test_routine(metrics, batch_size):

    # Initialize the callback
    callback = PINAProgressBar(metrics=metrics)

    # Convert to list if a single string is provided
    if isinstance(metrics, str):
        metrics = [metrics]

    # Convert to list if a single string is provided
    if isinstance(metrics, str):
        metrics = [metrics]

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

    # Get the expected metrics based on the input and batch size
    suffix = "_loss_epoch" if batch_size is not None else "_loss"
    expected_metrics = sorted([metric + suffix for metric in metrics])

    # Check that the progress bar metrics are the expected ones
    assert callback._sorted_metrics == expected_metrics

    # Assert that metrics in the progress bar are subset of expected metrics
    displayed_metrics = {
        key
        for key in trainer.progress_bar_metrics
        if key in callback._sorted_metrics
    }
    assert displayed_metrics.issubset(set(expected_metrics))
