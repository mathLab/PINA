import pytest
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.loss import SelfAdaptiveWeighting
from pina.problem.zoo import Poisson2DSquareProblem


# Initialize problem and model
problem = Poisson2DSquareProblem()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))


@pytest.mark.parametrize("k", [10, 100, 1000])
def test_constructor(k):
    SelfAdaptiveWeighting(k=k)

    # Should fail if k is not an integer
    with pytest.raises(AssertionError):
        SelfAdaptiveWeighting(k=1.5)

    # Should fail if k is not > 0
    with pytest.raises(AssertionError):
        SelfAdaptiveWeighting(k=0)

    # Should fail if k is not > 0
    with pytest.raises(AssertionError):
        SelfAdaptiveWeighting(k=-3)


@pytest.mark.parametrize("k", [2, 3])
def test_train_aggregation(k):
    weighting = SelfAdaptiveWeighting(k=k)
    solver = PINN(problem=problem, model=model, weighting=weighting)
    trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    trainer.train()
