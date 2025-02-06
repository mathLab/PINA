import pytest
import torch

from pina import LabelTensor
from pina.model import FeedForward
from pina.trainer import Trainer
from pina.solvers import RBAPINN
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
@pytest.mark.parametrize("eta", [1, 0.001])
@pytest.mark.parametrize("gamma", [0.5, 0.9])
def test_constructor(problem, eta, gamma):
    with pytest.raises(AssertionError):
        rba_pinn = RBAPINN(model=model, problem=problem, gamma=1.5)
    rba_pinn = RBAPINN(model=model, problem=problem, eta=eta, gamma=gamma)

    assert rba_pinn.accepted_conditions_types == (
        InputOutputPointsCondition,
        InputPointsEquationCondition,
        DomainEquationCondition
    )


@pytest.mark.parametrize("problem", [poisson_problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_rba_pinn_train(problem, batch_size):
    rba_pinn = RBAPINN(model=model, problem=problem)
    trainer = Trainer(solver=rba_pinn,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=1.,
                      val_size=0.,
                      test_size=0.)
    trainer.train()


@pytest.mark.parametrize("problem", [poisson_problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_rba_pinn_validation(problem, batch_size):
    rba_pinn = RBAPINN(model=model, problem=problem)
    trainer = Trainer(solver=rba_pinn,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=0.9,
                      val_size=0.1,
                      test_size=0.)
    trainer.train()


@pytest.mark.parametrize("problem", [poisson_problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_rba_pinn_test(problem, batch_size):
    rba_pinn = RBAPINN(model=model, problem=problem)
    trainer = Trainer(solver=rba_pinn,
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
    rba_pinn = RBAPINN(model=model, problem=problem)
    trainer = Trainer(solver=rba_pinn,
                      max_epochs=5,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.7,
                      val_size=0.2,
                      test_size=0.1,
                      default_root_dir=dir)
    trainer.train()

    # restore
    new_trainer = Trainer(solver=rba_pinn, max_epochs=5, accelerator='cpu')
    new_trainer.train(
        ckpt_path=f'{dir}/lightning_logs/version_0/checkpoints/' +
                   'epoch=4-step=5.ckpt')

    # loading
    new_solver = RBAPINN.load_from_checkpoint(
        f'{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt',
        problem=problem, model=model)

    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 1)
    assert new_solver.forward(test_pts).shape == (
        rba_pinn.forward(test_pts).shape
    )
    torch.testing.assert_close(
        new_solver.forward(test_pts),
        rba_pinn.forward(test_pts))

    # rm directories
    import shutil
    shutil.rmtree('tests/test_solvers/tmp')
