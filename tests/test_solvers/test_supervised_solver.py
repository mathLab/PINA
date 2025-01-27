import torch
import pytest
from pina.problem import AbstractProblem
from pina.problem.zoo import Poisson2DSquareProblem
from pina import Condition, LabelTensor
from pina.solvers import SupervisedSolver
from pina.model import FeedForward
from pina.trainer import Trainer


class FooProblem(AbstractProblem):
    input_variables = ['u_0', 'u_1']
    output_variables = ['u']
    conditions = {
        'data': Condition(
            input_points=LabelTensor(torch.tensor([[0., 1.]]), ['u_0', 'u_1']),
            output_points=LabelTensor(torch.tensor([[0.]]), ['u'])),
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
        return LabelTensor(t, ['u_0u_1'])


problem = FooProblem()
extra_feats = [myFeature()]
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
model_extra_feats = FeedForward(
    len(problem.input_variables) + 1, len(problem.output_variables))



def test_constructor():
    SupervisedSolver(problem=problem, model=model)

def test_wrong_constructor():
    with pytest.raises(ValueError):
        SupervisedSolver(problem=Poisson2DSquareProblem(), model=model)

def test_train_batch_size_full():
    solver = SupervisedSolver(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=1.,
                      test_size=0.,
                      val_size=0.)
    trainer.train() 
    solver = SupervisedSolver(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=None,
                      train_size=1.,
                      test_size=0.,
                      val_size=0.) 
    trainer.train()  
 
def test_train_and_val_cpu():

    solver = SupervisedSolver(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=5,
                      train_size=0.9,
                      test_size=0.1,
                      val_size=0.)
    trainer.train()

def test_train_and_val_gpu():
    solver = SupervisedSolver(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='mps',
                      batch_size=5,
                      train_size=1,
                      test_size=0.,
                      val_size=0.)
    trainer.train()

def test_extra_features_constructor():
    SupervisedSolver(problem=problem,
                     model=model_extra_feats,
                     extra_features=extra_feats)

def test_extra_features_train_and_val_cpu():
    solver = SupervisedSolver(problem=problem,
                              model=model_extra_feats,
                              extra_features=extra_feats)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='cpu',
                      batch_size=5,
                      train_size=0.9,
                      test_size=0.1,
                      )
    trainer.train()

def test_extra_features_train_and_val_gpu():
    solver = SupervisedSolver(problem=problem,
                              model=model_extra_feats,
                              extra_features=extra_feats)
    trainer = Trainer(solver=solver,
                      max_epochs=2,
                      accelerator='mps',
                      batch_size=5,
                      train_size=0.9,
                      test_size=0.1,
                      )
    trainer.train()
