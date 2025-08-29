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


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_constructor(alpha):
    NeuralTangentKernelWeighting(alpha=alpha)

    # Should fail if alpha is not >= 0
    with pytest.raises(ValueError):
        NeuralTangentKernelWeighting(alpha=-0.1)

    # Should fail if alpha is not <= 1
    with pytest.raises(ValueError):
        NeuralTangentKernelWeighting(alpha=1.1)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_train_aggregation(alpha):
    weighting = NeuralTangentKernelWeighting(alpha=alpha)
    solver = PINN(problem=problem, model=model, weighting=weighting)
    trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    trainer.train()
