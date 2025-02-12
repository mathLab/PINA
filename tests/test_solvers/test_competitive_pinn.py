import torch
import pytest 

from pina import LabelTensor
from pina.solvers import CompetitivePINN as CompPINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import (
    Poisson2DSquareProblem as Poisson,
    InversePoisson2DSquareProblem as InversePoisson
)
from pina.condition import (
    InputOutputPointsCondition,
    InputPointsEquationCondition,
    DomainEquationCondition
)


# make the problem
problem = Poisson()
problem.discretise_domain(100)
inverse_problem = InversePoisson()
inverse_problem.discretise_domain(100)
model = FeedForward(len(problem.input_variables), len(problem.output_variables))

@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("discr", [None, model])
def test_constructor(problem, discr):
    solver = CompPINN(problem=problem, model=model)
    solver = CompPINN(problem=problem, model=model, discriminator=discr)

    assert solver.accepted_conditions_types == (
        InputOutputPointsCondition,
        InputPointsEquationCondition,
        DomainEquationCondition
    )

@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_solver_train(problem, batch_size):
    solver = CompPINN(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=1.,
                      val_size=0.,
                      test_size=0.)
    trainer.train()



@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_solver_validation(problem, batch_size):
    solver = CompPINN(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=0.9,
                      val_size=0.1,
                      test_size=0.)
    trainer.train()

@pytest.mark.parametrize("problem", [problem, inverse_problem])
@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
def test_solver_test(problem, batch_size):
    solver = CompPINN(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=0.7,
                      val_size=0.2,
                      test_size=0.1)
    trainer.test()

@pytest.mark.parametrize("problem", [problem, inverse_problem])
def test_train_load_restore(problem):
    dir = "tests/test_solvers/tmp"
    problem = problem
    solver = CompPINN(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=5,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.7,
                      val_size=0.2,
                      test_size=0.1,
                      default_root_dir=dir)
    trainer.train()

    # restore
    new_trainer = Trainer(solver=solver, max_epochs=5, accelerator='cpu')
    new_trainer.train(
        ckpt_path=f'{dir}/lightning_logs/version_0/checkpoints/' +
                   'epoch=4-step=5.ckpt')

    # loading
    new_solver = CompPINN.load_from_checkpoint(
        f'{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt',
        problem=problem, model=model)

    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 1)
    assert new_solver.forward(test_pts).shape == (
        solver.forward(test_pts).shape
    )
    torch.testing.assert_close(
        new_solver.forward(test_pts),
        solver.forward(test_pts))

    # rm directories
    import shutil
    shutil.rmtree('tests/test_solvers/tmp')