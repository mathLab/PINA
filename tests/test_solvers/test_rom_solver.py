import torch
import pytest

from pina.problem import AbstractProblem
from pina import Condition, LabelTensor
from pina.solvers import ReducedOrderModelSolver
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.loss import LpLoss
from pina.problem.zoo import Poisson2DSquareProblem


class FooProblem(AbstractProblem):
    input_variables = ['u_0', 'u_1']
    output_variables = ['u']
    conditions = {
        'data': Condition(
            input_points=LabelTensor(torch.tensor([[0., 1.]]), ['u_0', 'u_1']),
            output_points=LabelTensor(torch.tensor([[0.]]), ['u'])),
    }

class myFeature(torch.nn.Module):
    def __init__(self):
        super(myFeature, self).__init__()
    def forward(self, x):
        t = (torch.sin(x.extract(['u_0']) * torch.pi) *
             torch.sin(x.extract(['u_1']) * torch.pi))
        return LabelTensor(t, ['u_0u_1'])


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
problem = FooProblem()
interpolation_net = FeedForward(len(problem.input_variables),
                                rank)
reduction_net = AE(len(problem.output_variables), rank)


def test_constructor():
    ReducedOrderModelSolver(problem=problem,reduction_network=reduction_net,
    interpolation_network=interpolation_net)

def test_wrong_constructor():
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

def test_train_batch_size_full():
    solver = ReducedOrderModelSolver(problem=problem,
                                     reduction_network=reduction_net,
                                     interpolation_network=interpolation_net)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=1.,
                      test_size=0.,
                      val_size=0.)
    trainer.train()  
 
def test_train_and_val_cpu():
    solver = ReducedOrderModelSolver(problem=problem,
                                     reduction_network=reduction_net,
                                     interpolation_network=interpolation_net)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=5,
                      train_size=0.9,
                      test_size=0.1,
                      val_size=0.)
    trainer.train()

# def test_train_and_val_gpu():
#     solver = ReducedOrderModelSolver(problem=problem,
#                                      reduction_network=reduction_net,
#                                      interpolation_network=interpolation_net)
#     trainer = Trainer(solver=solver,
#                       max_epochs=2,
#                       accelerator='gpu',
#                       batch_size=5,
#                       train_size=0.9,
#                       test_size=0.1,
#                       val_size=0.)
#     trainer.train()

def test_train_restore():
    tmpdir = "tests/test_solvers/tmp/tmp_restore"
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
                      default_root_dir=tmpdir)
    trainer.train()
    ntrainer = Trainer(solver=solver,
                       max_epochs=5,
                       accelerator='cpu',)
    ntrainer.train(
        ckpt_path=f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt')
    import shutil
    shutil.rmtree('tests/test_solvers/tmp')


def test_train_load():
    tmpdir = "tests/test_solvers/tmp/tmp_load"
    solver = ReducedOrderModelSolver(problem=problem,
               reduction_network=reduction_net,
               interpolation_network=interpolation_net)
    trainer = Trainer(solver=solver,
                      max_epochs=5,
                      accelerator='cpu',
                      batch_size=None,
                    #   train_size=0.9,
                    #   test_size=0.1,
                    #   val_size=0.,
                      default_root_dir=tmpdir)
    trainer.train()
    new_solver = ReducedOrderModelSolver.load_from_checkpoint(
        f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt',
        problem = problem,reduction_network=reduction_net,
        interpolation_network=interpolation_net)
    test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
    assert new_solver.forward(test_pts).shape == (20, 1)
    assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
    torch.testing.assert_close(
        new_solver.forward(test_pts),
        solver.forward(test_pts))
    import shutil
    shutil.rmtree('tests/test_solvers/tmp')