import pytest
from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem
from pina.loss import NeuralTangetKernelWeighting

problem = Poisson2DSquareProblem()
condition_names = problem.conditions.keys()


@pytest.mark.parametrize(
    "model,alpha",
    [
        (
            FeedForward(
                len(problem.input_variables), len(problem.output_variables)
            ),
            0.5,
        )
    ],
)
def test_constructor(model, alpha):
    NeuralTangetKernelWeighting(model=model, alpha=alpha)


@pytest.mark.parametrize(
    "model,alpha",
    [
        (
            FeedForward(
                len(problem.input_variables), len(problem.output_variables)
            ),
            1.2,
        )
    ],
)
def test_wrong_constructor2(model, alpha):
    with pytest.raises(ValueError):
        NeuralTangetKernelWeighting(model, alpha)


@pytest.mark.parametrize("model,alpha", [(0.5, 0.5)])
def test_wrong_constructor1(model):
    with pytest.raises(TypeError):
        NeuralTangetKernelWeighting(model)


@pytest.mark.parametrize(
    "model,alpha",
    [
        (
            FeedForward(
                len(problem.input_variables), len(problem.output_variables)
            ),
            0.5,
        )
    ],
)
def test_train_aggregation(model, alpha):
    weighting = NeuralTangetKernelWeighting(model=model, alpha=alpha)
    problem.discretise_domain(50)
    solver = PINN(problem=problem, model=model, weighting=weighting)
    trainer = Trainer(solver=solver, max_epochs=5, accelerator="cpu")
    trainer.train()
