import torch

from pina.problem import AbstractProblem
from pina import Condition, LabelTensor
from pina.solvers import SupervisedSolver
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.loss import LpLoss


class NeuralOperatorProblem(AbstractProblem):
    input_variables = ['u_0', 'u_1']
    output_variables = ['u']
    conditions = {
        # 'data' : Condition(
        #     input_points=LabelTensor(torch.rand(100, 2), input_variables), 
        #     output_points=LabelTensor(torch.rand(100, 1), output_variables))
    }

class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (torch.sin(x.extract(['u_0']) * torch.pi) *
             torch.sin(x.extract(['u_1']) * torch.pi))
        return LabelTensor(t, ['sin(x)sin(y)'])


# make the problem + extra feats
problem = NeuralOperatorProblem()
extra_feats = [myFeature()]
model = FeedForward(len(problem.input_variables),
                    len(problem.output_variables))
model_extra_feats = FeedForward(
    len(problem.input_variables) + 1,
    len(problem.output_variables))


def test_constructor():
    SupervisedSolver(problem=problem, model=model)


# def test_constructor_extra_feats():
#     SupervisedSolver(problem=problem, model=model_extra_feats, extra_features=extra_feats)


def test_train_cpu():
    solver = SupervisedSolver(problem = problem, model=model, loss=LpLoss())
    trainer = Trainer(solver=solver, max_epochs=3, accelerator='cpu', batch_size=20)
    trainer.train()


# def test_train_restore():
#     tmpdir = "tests/tmp_restore"
#     solver = SupervisedSolver(problem=problem,
#                 model=model,
#                 extra_features=None,
#                 loss=LpLoss())
#     trainer = Trainer(solver=solver,
#                       max_epochs=5,
#                       accelerator='cpu',
#                       default_root_dir=tmpdir)
#     trainer.train()
#     ntrainer = Trainer(solver=solver, max_epochs=15, accelerator='cpu')
#     t = ntrainer.train(
#         ckpt_path=f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt')
#     import shutil
#     shutil.rmtree(tmpdir)


# def test_train_load():
#     tmpdir = "tests/tmp_load"
#     solver = SupervisedSolver(problem=problem,
#                 model=model,
#                 extra_features=None,
#                 loss=LpLoss())
#     trainer = Trainer(solver=solver,
#                       max_epochs=15,
#                       accelerator='cpu',
#                       default_root_dir=tmpdir)
#     trainer.train()
#     new_solver = SupervisedSolver.load_from_checkpoint(
#         f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=14-step=15.ckpt',
#         problem = problem, model=model)
#     test_pts = LabelTensor(torch.rand(20, 2), problem.input_variables)
#     assert new_solver.forward(test_pts).shape == (20, 1)
#     assert new_solver.forward(test_pts).shape == solver.forward(test_pts).shape
#     torch.testing.assert_close(
#         new_solver.forward(test_pts),
#         solver.forward(test_pts))
#     import shutil
#     shutil.rmtree(tmpdir)

# def test_train_extra_feats_cpu():
#     pinn = SupervisedSolver(problem=problem,
#                 model=model_extra_feats,
#                 extra_features=extra_feats)
#     trainer = Trainer(solver=pinn, max_epochs=5, accelerator='cpu')
#     trainer.train()