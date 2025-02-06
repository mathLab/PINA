import pytest
import torch

from pina import LabelTensor
from pina.model import FeedForward
from pina.trainer import Trainer
from pina.solvers import PINN
from pina.condition import (
    InputOutputPointsCondition,
    InputPointsEquationCondition,
    DomainEquationCondition
)
from pina.problem.zoo import (
    Poisson2DSquareProblem as Poisson,
    InversePoisson2DSquareProblem as InversePoisson
)


# define problems and model
poisson_problem = Poisson()
poisson_problem.discretise_domain(100)
inverse_problem = InversePoisson()
inverse_problem.discretise_domain(100)
model = FeedForward(
    len(poisson_problem.input_variables),
    len(poisson_problem.output_variables)
)


@pytest.mark.parametrize("problem", [poisson_problem, inverse_problem])
def test_constructor(problem):
    pinn = PINN(problem=problem, model=model)

    assert pinn.accepted_conditions_types == (
        InputOutputPointsCondition,
        InputPointsEquationCondition,
        DomainEquationCondition
    )

@pytest.mark.parametrize("problem", [poisson_problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_pinn_train(problem, batch_size):
    pinn = PINN(model=model, problem=problem)
    trainer = Trainer(solver=pinn,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=1.,
                      val_size=0.,
                      test_size=0.)
    trainer.train()


@pytest.mark.parametrize("problem", [poisson_problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_pinn_validation(problem, batch_size):
    pinn = PINN(model=model, problem=problem)
    trainer = Trainer(solver=pinn,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=0.9,
                      val_size=0.1,
                      test_size=0.)
    trainer.train()


@pytest.mark.parametrize("problem", [poisson_problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_pinn_test(problem, batch_size):
    pinn = PINN(model=model, problem=problem)
    trainer = Trainer(solver=pinn,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=0.7,
                      val_size=0.2,
                      test_size=0.1)
    trainer.test()


@pytest.mark.parametrize("problem", [poisson_problem, inverse_problem])
def test_train_load_restore(problem):
    dir = "tests/test_solvers/tmp"
    problem = problem
    pinn = PINN(model=model, problem=problem)
    trainer = Trainer(solver=pinn,
                      max_epochs=5,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.7,
                      val_size=0.2,
                      test_size=0.1,
                      default_root_dir=dir)
    trainer.train()

    # restore
    new_trainer = Trainer(solver=pinn, max_epochs=5, accelerator='cpu')
    new_trainer.train(
        ckpt_path=f'{dir}/lightning_logs/version_0/checkpoints/' +
                   'epoch=4-step=5.ckpt')

    # loading
    new_solver = PINN.load_from_checkpoint(
        f'{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt',
        problem=problem, model=model)

    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 1)
    assert new_solver.forward(test_pts).shape == pinn.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts),
        pinn.forward(test_pts))

    # rm directories
    import shutil
    shutil.rmtree('tests/test_solvers/tmp')
