import pytest

from torch.nn import MSELoss

from pina.solver import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem as Poisson
from pina.callback.refinement import R3Refinement


# make the problem
poisson_problem = Poisson()
poisson_problem.discretise_domain(10, "grid", domains=["g1", "g2", "g3", "g4"])
poisson_problem.discretise_domain(10, "grid", domains="D")
model = FeedForward(
    len(poisson_problem.input_variables), len(poisson_problem.output_variables)
)
solver = PINN(problem=poisson_problem, model=model)


def test_constructor():
    # good constructor
    R3Refinement(sample_every=10)
    R3Refinement(sample_every=10, residual_loss=MSELoss)
    R3Refinement(sample_every=10, condition_to_update=["D"])
    # wrong constructor
    with pytest.raises(ValueError):
        R3Refinement(sample_every="str")
    with pytest.raises(ValueError):
        R3Refinement(sample_every=10, condition_to_update=3)


def test_sample():
    trainer = Trainer(
        solver=solver,
        callbacks=[R3Refinement(sample_every=1)],
        accelerator="cpu",
        max_epochs=5,
    )
    before_n_points = {
        loc: len(pts) for loc, pts in trainer.solver.problem.input_pts.items()
    }
    trainer.train()
    after_n_points = {
        loc: len(pts)
        for loc, pts in trainer.data_module.train_dataset.input.items()
    }
    assert before_n_points == trainer.callbacks[0].initial_population_size
    assert before_n_points == after_n_points
