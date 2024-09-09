import torch
import pytest

from pina.problem import AbstractProblem
from pina import Condition, LabelTensor
from pina.solvers import ReducedOrderModelSolver
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.loss.loss_interface import LpLoss


class NeuralOperatorProblem(AbstractProblem):
    input_variables = ['u_0', 'u_1']
    output_variables = [f'u_{i}' for i in range(100)]
    conditions = {'data' : Condition(input_points=
                                     LabelTensor(torch.rand(10, 2),
                                                 input_variables), 
                                     output_points=
                                     LabelTensor(torch.rand(10, 100),
                                                 output_variables))}


# make the problem + extra feats
class AE(torch.nn.Module):
    def __init__(self, input_dimensions, rank):
        super().__init__()
        self.encode = FeedForward(input_dimensions, rank, layers=[input_dimensions//4])
        self.decode = FeedForward(rank, input_dimensions, layers=[input_dimensions//4])
class AE_missing_encode(torch.nn.Module):
    def __init__(self, input_dimensions, rank):
        super().__init__()
        self.encode = FeedForward(input_dimensions, rank, layers=[input_dimensions//4])
class AE_missing_decode(torch.nn.Module):
    def __init__(self, input_dimensions, rank):
        super().__init__()
        self.decode = FeedForward(rank, input_dimensions, layers=[input_dimensions//4])

rank = 10
problem = NeuralOperatorProblem()
interpolation_net = FeedForward(len(problem.input_variables),
                                rank)
reduction_net = AE(len(problem.output_variables), rank)

def test_constructor():
    ReducedOrderModelSolver(problem=problem,reduction_network=reduction_net,
    interpolation_network=interpolation_net)
    with pytest.raises(SyntaxError):
        ReducedOrderModelSolver(problem=problem,
                     reduction_network=AE_missing_encode(
                         len(problem.output_variables), rank),
                     interpolation_network=interpolation_net)
        ReducedOrderModelSolver(problem=problem,
                     reduction_network=AE_missing_decode(
                         len(problem.output_variables), rank),
                     interpolation_network=interpolation_net)


def test_train_cpu():
    solver = ReducedOrderModelSolver(problem = problem,reduction_network=reduction_net,
    interpolation_network=interpolation_net, loss=LpLoss())
    trainer = Trainer(solver=solver, max_epochs=3, accelerator='cpu', batch_size=20)
    trainer.train()


def test_train_restore():
    tmpdir = "tests/tmp_restore"
    solver = ReducedOrderModelSolver(problem=problem,
               reduction_network=reduction_net,
               interpolation_network=interpolation_net,
               loss=LpLoss())
    trainer = Trainer(solver=solver,
                      max_epochs=5,
                      accelerator='cpu',
                      default_root_dir=tmpdir)
    trainer.train()
    ntrainer = Trainer(solver=solver, max_epochs=15, accelerator='cpu')
    t = ntrainer.train(
        ckpt_path=f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt')
    import shutil
    shutil.rmtree(tmpdir)


def test_train_load():
    tmpdir = "tests/tmp_load"
    solver = ReducedOrderModelSolver(problem=problem,
               reduction_network=reduction_net,
               interpolation_network=interpolation_net,
               loss=LpLoss())
    trainer = Trainer(solver=solver,
                      max_epochs=15,
                      accelerator='cpu',
                      default_root_dir=tmpdir)
    trainer.train()
    new_solver = ReducedOrderModelSolver.load_from_checkpoint(
        f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=14-step=15.ckpt',
        problem = problem,reduction_network=reduction_net,
        interpolation_network=interpolation_net)
    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 100)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts),
        solver.forward(test_pts))
    import shutil
    shutil.rmtree(tmpdir)