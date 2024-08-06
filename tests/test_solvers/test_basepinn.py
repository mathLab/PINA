import torch
import pytest

from pina import Condition, LabelTensor, Trainer
from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina.geometry import CartesianDomain
from pina.model import FeedForward
from pina.solvers import PINNInterface
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue

def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                  torch.sin(input_.extract(['y']) * torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term


my_laplace = Equation(laplace_equation)
in_ = LabelTensor(torch.tensor([[0., 1.]]), ['x', 'y'])
out_ = LabelTensor(torch.tensor([[0.]]), ['u'])
in2_ = LabelTensor(torch.rand(60, 2), ['x', 'y'])
out2_ = LabelTensor(torch.rand(60, 1), ['u'])



class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x': [0, 1], 'y':  1}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': 0}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x':  1, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': 0, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'D': Condition(
            input_points=LabelTensor(torch.rand(size=(100, 2)), ['x', 'y']),
            equation=my_laplace),
        'data': Condition(
            input_points=in_,
            output_points=out_),
        'data2': Condition(
            input_points=in2_,
            output_points=out2_)
    }

    def poisson_sol(self, pts):
        return -(torch.sin(pts.extract(['x']) * torch.pi) *
                 torch.sin(pts.extract(['y']) * torch.pi)) / (2 * torch.pi**2)

    truth_solution = poisson_sol

class FOOPINN(PINNInterface):
    def __init__(self, model, problem):
        super().__init__(models=[model], problem=problem,
                         optimizers=[torch.optim.Adam],
                         optimizers_kwargs=[{'lr' : 0.001}],
                         extra_features=None,
                         loss=torch.nn.MSELoss())
    def forward(self, x):
        return self.models[0](x)

    def loss_phys(self, samples, equation):
        residual = self.compute_residual(samples=samples, equation=equation)
        loss_value = self.loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        self.store_log(loss_value=float(loss_value))
        return loss_value

    def configure_optimizers(self):
        return self.optimizers, []

# make the problem
poisson_problem = Poisson()
poisson_problem.discretise_domain(100)
model = FeedForward(len(poisson_problem.input_variables),
                    len(poisson_problem.output_variables))
model_extra_feats = FeedForward(
    len(poisson_problem.input_variables) + 1,
    len(poisson_problem.output_variables))


def test_constructor():
    with pytest.raises(TypeError):
        PINNInterface()
    # a simple pinn built with PINNInterface
    FOOPINN(model, poisson_problem)

def test_train_step():
    solver = FOOPINN(model, poisson_problem)
    trainer = Trainer(solver, max_epochs=2, accelerator='cpu')
    trainer.train()

def test_log():
    solver = FOOPINN(model, poisson_problem)
    trainer = Trainer(solver, max_epochs=2, accelerator='cpu')
    trainer.train()
    # assert the logged metrics are correct
    logged_metrics = sorted(list(trainer.logged_metrics.keys()))
    total_metrics = sorted(
        list([key + '_loss' for key in poisson_problem.conditions.keys()])
        + ['mean_loss'])
    assert logged_metrics == total_metrics