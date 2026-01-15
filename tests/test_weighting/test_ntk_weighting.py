import pytest
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.loss import NeuralTangentKernelWeighting
from pina.problem.zoo import Poisson2DSquareProblem


# Initialize problem and model
problem = Poisson2DSquareProblem()
problem.discretise_domain(10)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))


@pytest.mark.parametrize("update_every_n_epochs", [1, 10, 100, 1000])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_constructor(update_every_n_epochs, alpha):
    NeuralTangentKernelWeighting(
        update_every_n_epochs=update_every_n_epochs, alpha=alpha
    )

    # Should fail if alpha is not >= 0
    with pytest.raises(ValueError):
        NeuralTangentKernelWeighting(
            update_every_n_epochs=update_every_n_epochs, alpha=-0.1
        )

    # Should fail if alpha is not <= 1
    with pytest.raises(ValueError):
        NeuralTangentKernelWeighting(alpha=1.1)

    # Should fail if update_every_n_epochs is not an integer
    with pytest.raises(AssertionError):
        NeuralTangentKernelWeighting(update_every_n_epochs=1.5)

    # Should fail if update_every_n_epochs is not > 0
    with pytest.raises(AssertionError):
        NeuralTangentKernelWeighting(update_every_n_epochs=0)

    # Should fail if update_every_n_epochs is not > 0
    with pytest.raises(AssertionError):
        NeuralTangentKernelWeighting(update_every_n_epochs=-3)


@pytest.mark.parametrize("update_every_n_epochs", [1, 3])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_train_aggregation(update_every_n_epochs, alpha):
    weighting = NeuralTangentKernelWeighting(
        update_every_n_epochs=update_every_n_epochs, alpha=alpha
    )
    solver = PINN(problem=problem, model=model, weighting=weighting)
    trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    trainer.train()
