import torch
import pytest
from pina.problem import AbstractProblem, SpatialProblem
from pina import Condition, LabelTensor
from pina.solvers import SupervisedSolver
from pina.model import FeedForward
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue
from pina.operators import laplacian
from pina.domain import CartesianDomain
from pina.trainer import Trainer

in_ = LabelTensor(torch.tensor([[0., 1.]]), ['u_0', 'u_1'])
out_ = LabelTensor(torch.tensor([[0.]]), ['u'])


class NeuralOperatorProblem(AbstractProblem):
    input_variables = ['u_0', 'u_1']
    output_variables = ['u']

    conditions = {
        'data': Condition(input_points=in_, output_points=out_),
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


problem = NeuralOperatorProblem()
extra_feats = [myFeature()]
model = FeedForward(len(problem.input_variables), len(problem.output_variables))
model_extra_feats = FeedForward(
    len(problem.input_variables) + 1, len(problem.output_variables))


def test_constructor():
    SupervisedSolver(problem=problem, model=model)


test_constructor()


def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                  torch.sin(input_.extract(['y']) * torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term


my_laplace = Equation(laplace_equation)


class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'gamma1':
            Condition(domain=CartesianDomain({
                'x': [0, 1],
                'y': 1
            }),
                equation=FixedValue(0.0)),
        'gamma2':
            Condition(domain=CartesianDomain({
                'x': [0, 1],
                'y': 0
            }),
                equation=FixedValue(0.0)),
        'gamma3':
            Condition(domain=CartesianDomain({
                'x': 1,
                'y': [0, 1]
            }),
                equation=FixedValue(0.0)),
        'gamma4':
            Condition(domain=CartesianDomain({
                'x': 0,
                'y': [0, 1]
            }),
                equation=FixedValue(0.0)),
        'D':
            Condition(domain=CartesianDomain({
                'x': [0, 1],
                'y': [0, 1]
            }),
                equation=my_laplace),
        'data':
            Condition(input_points=in_, output_points=out_)
    }

    def poisson_sol(self, pts):
        return -(torch.sin(pts.extract(['x']) * torch.pi) *
                 torch.sin(pts.extract(['y']) * torch.pi)) / (2 * torch.pi ** 2)

    truth_solution = poisson_sol


def test_wrong_constructor():
    poisson_problem = Poisson()
    with pytest.raises(ValueError):
        SupervisedSolver(problem=poisson_problem, model=model)


def test_train_cpu():
    solver = SupervisedSolver(problem=problem, model=model)
    trainer = Trainer(solver=solver,
                      max_epochs=200,
                      accelerator='gpu',
                      batch_size=5,
                      train_size=1,
                      test_size=0.,
                      eval_size=0.)
    trainer.train()
test_train_cpu()


def test_extra_features_constructor():
    SupervisedSolver(problem=problem,
                     model=model_extra_feats,
                     extra_features=extra_feats)


def test_extra_features_train_cpu():
    solver = SupervisedSolver(problem=problem,
                              model=model_extra_feats,
                              extra_features=extra_feats)
    trainer = Trainer(solver=solver,
                      max_epochs=200,
                      accelerator='gpu',
                      batch_size=5)
    trainer.train()
