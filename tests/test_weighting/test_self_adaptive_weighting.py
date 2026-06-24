import pytest
from pina import Trainer
from pina.solver import PhysicsInformedSingleModelSolver
from pina.model import FeedForward
from pina.weighting import SelfAdaptiveWeighting
from pina.problem.zoo import Poisson2DSquareProblem

# Initialize problem and model
problem = Poisson2DSquareProblem()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))


@pytest.mark.parametrize("update_every_n_epochs", [1, 10])
def test_constructor(update_every_n_epochs):
    SelfAdaptiveWeighting(update_every_n_epochs=update_every_n_epochs)

    # Should fail if update_every_n_epochs is not an integer
    with pytest.raises(AssertionError):
        SelfAdaptiveWeighting(update_every_n_epochs=1.5)

    # Should fail if update_every_n_epochs is not > 0
    with pytest.raises(AssertionError):
        SelfAdaptiveWeighting(update_every_n_epochs=0)


@pytest.mark.parametrize("update_every_n_epochs", [1, 3])
def test_aggregate(update_every_n_epochs):

    # Initialize weighting, solver, and trainer
    weighting = SelfAdaptiveWeighting(
        update_every_n_epochs=update_every_n_epochs
    )
    solver = PhysicsInformedSingleModelSolver(
        problem=problem, model=model, weighting=weighting
    )
    trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")

    # Train
    trainer.train()

    # Check that weights keys are the same as loss keys
    assert weighting.last_saved_weights().keys() == problem.conditions.keys()
