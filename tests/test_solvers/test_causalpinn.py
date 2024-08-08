import torch
import pytest

from pina.problem import TimeDependentProblem, InverseProblem, SpatialProblem
from pina.operators import grad
from pina.domain import CartesianDomain
from pina import Condition, LabelTensor
from pina.solvers import CausalPINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue
from pina.loss import LpLoss



class FooProblem(SpatialProblem):
    '''
    Foo problem formulation.
    '''
    output_variables = ['u']
    conditions = {}
    spatial_domain = None


class InverseDiffusionReactionSystem(TimeDependentProblem, SpatialProblem, InverseProblem):

    def diffusionreaction(input_, output_, params_):
        x = input_.extract('x')
        t = input_.extract('t')
        u_t = grad(output_, input_, d='t')
        u_x = grad(output_, input_, d='x')
        u_xx = grad(u_x, input_, d='x')
        r = torch.exp(-t) * (1.5 * torch.sin(2*x) + (8/3)*torch.sin(3*x) +
                             (15/4)*torch.sin(4*x) + (63/8)*torch.sin(8*x))
        return u_t - params_['mu']*u_xx - r

    def _solution(self, pts):
        t = pts.extract('t')
        x = pts.extract('x')
        return torch.exp(-t) * (torch.sin(x) + (1/2)*torch.sin(2*x) +
                                (1/3)*torch.sin(3*x) + (1/4)*torch.sin(4*x) +
                                (1/8)*torch.sin(8*x))
    
    # assign output/ spatial and temporal variables
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-torch.pi, torch.pi]})
    temporal_domain = CartesianDomain({'t': [0, 1]})
    unknown_parameter_domain = CartesianDomain({'mu': [-1, 1]})

    # problem condition statement
    conditions = {
        'D': Condition(location=CartesianDomain({'x': [-torch.pi, torch.pi],
                                                 't': [0, 1]}),
                       equation=Equation(diffusionreaction)),
        'data' : Condition(input_points=LabelTensor(torch.tensor([[0., 0.]]), ['x', 't']),
                       output_points=LabelTensor(torch.tensor([[0.]]), ['u'])),
    }

class DiffusionReactionSystem(TimeDependentProblem, SpatialProblem):

    def diffusionreaction(input_, output_):
        x = input_.extract('x')
        t = input_.extract('t')
        u_t = grad(output_, input_, d='t')
        u_x = grad(output_, input_, d='x')
        u_xx = grad(u_x, input_, d='x')
        r = torch.exp(-t) * (1.5 * torch.sin(2*x) + (8/3)*torch.sin(3*x) +
                             (15/4)*torch.sin(4*x) + (63/8)*torch.sin(8*x))
        return u_t - u_xx - r

    def _solution(self, pts):
        t = pts.extract('t')
        x = pts.extract('x')
        return torch.exp(-t) * (torch.sin(x) + (1/2)*torch.sin(2*x) +
                                (1/3)*torch.sin(3*x) + (1/4)*torch.sin(4*x) +
                                (1/8)*torch.sin(8*x))
    
    # assign output/ spatial and temporal variables
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-torch.pi, torch.pi]})
    temporal_domain = CartesianDomain({'t': [0, 1]})

    # problem condition statement
    conditions = {
        'D': Condition(location=CartesianDomain({'x': [-torch.pi, torch.pi],
                                                 't': [0, 1]}),
                       equation=Equation(diffusionreaction)),
    }

class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (torch.sin(x.extract(['x']) * torch.pi))
        return LabelTensor(t, ['sin(x)'])


# make the problem
problem = DiffusionReactionSystem()
model = FeedForward(len(problem.input_variables),
                    len(problem.output_variables))
model_extra_feats = FeedForward(
    len(problem.input_variables) + 1,
    len(problem.output_variables))
extra_feats = [myFeature()]


def test_constructor():
    CausalPINN(problem=problem, model=model, extra_features=None)

    with pytest.raises(ValueError):
        CausalPINN(FooProblem(), model=model, extra_features=None)


def test_constructor_extra_feats():
    model_extra_feats = FeedForward(
        len(problem.input_variables) + 1,
        len(problem.output_variables))
    CausalPINN(problem=problem,
         model=model_extra_feats,
         extra_features=extra_feats)


def test_train_cpu():
    problem = DiffusionReactionSystem()
    boundaries = ['D']
    n = 10
    problem.discretise_domain(n, 'grid', locations=boundaries)
    pinn = CausalPINN(problem = problem,
                 model=model, extra_features=None, loss=LpLoss())
    trainer = Trainer(solver=pinn, max_epochs=1,
                      accelerator='cpu', batch_size=20)
    trainer.train()

def test_log():
    problem.discretise_domain(100)
    solver = CausalPINN(problem = problem,
                 model=model, extra_features=None, loss=LpLoss())
    trainer = Trainer(solver, max_epochs=2, accelerator='cpu')
    trainer.train()
    # assert the logged metrics are correct
    logged_metrics = sorted(list(trainer.logged_metrics.keys()))
    total_metrics = sorted(
        list([key + '_loss' for key in problem.conditions.keys()])
        + ['mean_loss'])
    assert logged_metrics == total_metrics

def test_train_restore():
    tmpdir = "tests/tmp_restore"
    problem = DiffusionReactionSystem()
    boundaries = ['D']
    n = 10
    problem.discretise_domain(n, 'grid', locations=boundaries)
    pinn = CausalPINN(problem=problem,
                model=model,
                extra_features=None,
                loss=LpLoss())
    trainer = Trainer(solver=pinn,
                      max_epochs=5,
                      accelerator='cpu',
                      default_root_dir=tmpdir)
    trainer.train()
    ntrainer = Trainer(solver=pinn, max_epochs=15, accelerator='cpu')
    t = ntrainer.train(
        ckpt_path=f'{tmpdir}/lightning_logs/version_0/'
        'checkpoints/epoch=4-step=5.ckpt')
    import shutil
    shutil.rmtree(tmpdir)


def test_train_load():
    tmpdir = "tests/tmp_load"
    problem = DiffusionReactionSystem()
    boundaries = ['D']
    n = 10
    problem.discretise_domain(n, 'grid', locations=boundaries)
    pinn = CausalPINN(problem=problem,
                model=model,
                extra_features=None,
                loss=LpLoss())
    trainer = Trainer(solver=pinn,
                      max_epochs=15,
                      accelerator='cpu',
                      default_root_dir=tmpdir)
    trainer.train()
    new_pinn = CausalPINN.load_from_checkpoint(
        f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=14-step=15.ckpt',
        problem = problem, model=model)
    test_pts = CartesianDomain({'x': [0, 1], 't': [0, 1]}).sample(10)
    assert new_pinn.forward(test_pts).extract(['u']).shape == (10, 1)
    assert new_pinn.forward(test_pts).extract(
        ['u']).shape == pinn.forward(test_pts).extract(['u']).shape
    torch.testing.assert_close(
        new_pinn.forward(test_pts).extract(['u']),
        pinn.forward(test_pts).extract(['u']))
    import shutil
    shutil.rmtree(tmpdir)

def test_train_inverse_problem_cpu():
    problem = InverseDiffusionReactionSystem()
    boundaries = ['D']
    n = 100
    problem.discretise_domain(n, 'random', locations=boundaries)
    pinn = CausalPINN(problem = problem,
                 model=model, extra_features=None, loss=LpLoss())
    trainer = Trainer(solver=pinn, max_epochs=1,
                      accelerator='cpu', batch_size=20)
    trainer.train()


# # TODO does not currently work
# def test_train_inverse_problem_restore():
#     tmpdir = "tests/tmp_restore_inv"
#     problem = InverseDiffusionReactionSystem()
#     boundaries = ['D']
#     n = 100
#     problem.discretise_domain(n, 'random', locations=boundaries)
#     pinn = CausalPINN(problem=problem,
#                 model=model,
#                 extra_features=None,
#                 loss=LpLoss())
#     trainer = Trainer(solver=pinn,
#                       max_epochs=5,
#                       accelerator='cpu',
#                       default_root_dir=tmpdir)
#     trainer.train()
#     ntrainer = Trainer(solver=pinn, max_epochs=5, accelerator='cpu')
#     t = ntrainer.train(
#         ckpt_path=f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=4-step=5.ckpt')
#     import shutil
#     shutil.rmtree(tmpdir)


def test_train_inverse_problem_load():
    tmpdir = "tests/tmp_load_inv"
    problem = InverseDiffusionReactionSystem()
    boundaries = ['D']
    n = 100
    problem.discretise_domain(n, 'random', locations=boundaries)
    pinn = CausalPINN(problem=problem,
                model=model,
                extra_features=None,
                loss=LpLoss())
    trainer = Trainer(solver=pinn,
                      max_epochs=15,
                      accelerator='cpu',
                      default_root_dir=tmpdir)
    trainer.train()
    new_pinn = CausalPINN.load_from_checkpoint(
        f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=14-step=30.ckpt',
        problem = problem, model=model)
    test_pts = CartesianDomain({'x': [0, 1], 't': [0, 1]}).sample(10)
    assert new_pinn.forward(test_pts).extract(['u']).shape == (10, 1)
    assert new_pinn.forward(test_pts).extract(
        ['u']).shape == pinn.forward(test_pts).extract(['u']).shape
    torch.testing.assert_close(
        new_pinn.forward(test_pts).extract(['u']),
        pinn.forward(test_pts).extract(['u']))
    import shutil
    shutil.rmtree(tmpdir)


def test_train_extra_feats_cpu():
    problem = DiffusionReactionSystem()
    boundaries = ['D']
    n = 10
    problem.discretise_domain(n, 'grid', locations=boundaries)
    pinn = CausalPINN(problem=problem,
                model=model_extra_feats,
                extra_features=extra_feats)
    trainer = Trainer(solver=pinn, max_epochs=5, accelerator='cpu')
    trainer.train()