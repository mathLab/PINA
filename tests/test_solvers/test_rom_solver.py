import torch
import pytest

from pina import Condition, LabelTensor
from pina.problem import AbstractProblem
from pina.condition import InputOutputPointsCondition
from pina.solvers import ReducedOrderModelSolver
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.problem.zoo import Poisson2DSquareProblem
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


class AE(torch.nn.Module):
    def __init__(self, input_dimensions, rank):
        super().__init__()
        self.encode = FeedForward(
            input_dimensions, rank, layers=[input_dimensions//4])
        self.decode = FeedForward(
            rank, input_dimensions, layers=[input_dimensions//4])


class AE_missing_encode(torch.nn.Module):
    def __init__(self, input_dimensions, rank):
        super().__init__()
        self.encode = FeedForward(
            input_dimensions, rank, layers=[input_dimensions//4])


class AE_missing_decode(torch.nn.Module):
    def __init__(self, input_dimensions, rank):
        super().__init__()
        self.decode = FeedForward(
            rank, input_dimensions, layers=[input_dimensions//4])


rank = 10
model = AE(2, 1)
interpolation_net = FeedForward(2, rank)
reduction_net = AE(1, rank)


def test_constructor():
    problem = TensorProblem()
    ReducedOrderModelSolver(problem=problem,
                            interpolation_network=interpolation_net,
                            reduction_network=reduction_net)
    ReducedOrderModelSolver(problem=LabelTensorProblem(),
                            reduction_network=reduction_net,
                            interpolation_network=interpolation_net)
    assert ReducedOrderModelSolver.accepted_conditions_types == InputOutputPointsCondition
    with pytest.raises(SyntaxError):
        ReducedOrderModelSolver(problem=problem,
                                reduction_network=AE_missing_encode(
                                    len(problem.output_variables), rank),
                                interpolation_network=interpolation_net)
        ReducedOrderModelSolver(problem=problem,
                                reduction_network=AE_missing_decode(
                                    len(problem.output_variables), rank),
                                interpolation_network=interpolation_net)
    with pytest.raises(ValueError):
        ReducedOrderModelSolver(problem=Poisson2DSquareProblem(),
                                reduction_network=reduction_net,
                                interpolation_network=interpolation_net)


@pytest.mark.parametrize("batch_size", [None, 1, 5, 20])
@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_train(use_lt, batch_size, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = ReducedOrderModelSolver(problem=problem,
                                     reduction_network=reduction_net,
                                     interpolation_network=interpolation_net, use_lt=use_lt)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=batch_size,
                      train_size=1.,
                      test_size=0.,
                      val_size=0.,
                      compile=compile)
    trainer.train()
    if trainer.compile:
        for v in solver.model.values():
            assert (isinstance(v, OptimizedModule))


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_validation(use_lt, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = ReducedOrderModelSolver(problem=problem,
                                     reduction_network=reduction_net,
                                     interpolation_network=interpolation_net, use_lt=use_lt)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.9,
                      val_size=0.1,
                      test_size=0.,
                      compile=compile)
    trainer.train()
    if trainer.compile:
        for v in solver.model.values():
            assert (isinstance(v, OptimizedModule))


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize("compile", [True, False])
def test_solver_test(use_lt, compile):
    problem = LabelTensorProblem() if use_lt else TensorProblem()
    solver = ReducedOrderModelSolver(problem=problem,
                                     reduction_network=reduction_net,
                                     interpolation_network=interpolation_net, use_lt=use_lt)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=0.8,
                      val_size=0.1,
                      test_size=0.1,
                      compile=compile)
    trainer.train()
    if trainer.compile:
        for v in solver.model.values():
            assert (isinstance(v, OptimizedModule))


def test_train_load_restore():
    dir = "tests/test_solvers/tmp/"
    problem = LabelTensorProblem()
    solver = ReducedOrderModelSolver(problem=problem,

                                     reduction_network=reduction_net,
                                     interpolation_network=interpolation_net)
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
    ntrainer = Trainer(solver=solver,
                       max_epochs=5,
                       accelerator='cpu',)
    ntrainer.train(
        ckpt_path=f'{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt')
    # loading
    new_solver = ReducedOrderModelSolver.load_from_checkpoint(
        f'{dir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt',
        problem=problem,
        reduction_network=reduction_net,
        interpolation_network=interpolation_net)
    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 1)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts),
        solver.forward(test_pts))
    # rm directories
    import shutil
    shutil.rmtree('tests/test_solvers/tmp')
