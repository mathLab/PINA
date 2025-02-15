import torch
import pytest
from pina import Condition, LabelTensor
from pina.condition import InputOutputPointsCondition
from pina.problem import AbstractProblem
from pina.solvers import SupervisedSolver
from pina.model import FeedForward
from pina.trainer import Trainer
from torch._dynamo.eval_frame import OptimizedModule


class LabelTensorProblem(AbstractProblem):
    input_variables = ['u_0', 'u_1']
    output_variables = ['u']
    conditions = {
        'data': Condition(
            input_points=LabelTensor(torch.randn(20, 2), ['u_0', 'u_1']),
            output_points=LabelTensor(torch.randn(20, 1), ['u'])),
    }


class TensorProblem(AbstractProblem):
    input_variables = ['u_0', 'u_1']
    output_variables = ['u']
    conditions = {
        'data': Condition(
            input_points=torch.randn(20, 2),
            output_points=torch.randn(20, 1))
    }


model = FeedForward(2, 1)


def test_constructor():
    SupervisedSolver(problem=TensorProblem(), model=model)
    SupervisedSolver(problem=LabelTensorProblem(), model=model)
    assert SupervisedSolver.accepted_conditions_types == (
        InputOutputPointsCondition
    )


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(use_lt, batch_size, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = SupervisedSolver(problem=problem, model=model, use_lt=use_lt)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=1.,
                      test_size=0.,
                      val_size=0.,
                      compile=compile)

    trainer.train()
    if compile:
        assert (isinstance(solver.model, OptimizedModule))


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_validation(use_lt, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = SupervisedSolver(problem=problem, model=model, use_lt=use_lt)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.9,
                      val_size=0.1,
                      test_size=0.,
                      compile=compile)
    trainer.train()
    if compile:
        assert (isinstance(solver.model, OptimizedModule))


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_test(use_lt, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = SupervisedSolver(problem=problem, model=model, use_lt=use_lt)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.8,
                      val_size=0.1,
                      test_size=0.1,
                      compile=compile)
    trainer.test()
    if compile:
        assert (isinstance(solver.model, OptimizedModule))


def test_train_load_restore():
    dir = "tests/test_solvers/tmp/"
    problem = LabelTensorProblem()
    solver = SupervisedSolver(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=5,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.9,
                      test_size=0.1,
                      val_size=0.,
                      default_root_dir=dir)
    trainer.train()

    # restore
    new_trainer = Trainer(solver=solver, max_epochs=5, accelerator='cpu')
    new_trainer.train(
        ckpt_path=f'{dir}/lightning_logs/version_0/checkpoints/' +
        'epoch=4-step=5.ckpt')

    # loading
    new_solver = SupervisedSolver.load_from_checkpoint(
        f'{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt',
        problem=problem, model=model)

    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 1)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts),
        solver.forward(test_pts))

    # rm directories
    import shutil
    shutil.rmtree('tests/test_solvers/tmp')
