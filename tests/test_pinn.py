import torch
import pytest

from pina import LabelTensor, Condition, Span, PINN
from pina.problem import SpatialProblem
from pina.model import FeedForward
from pina.operators import nabla

in_ = LabelTensor(torch.tensor([[0., 1.]]), ['x', 'y'])
out_ = LabelTensor(torch.tensor([[0.]]), ['u'])

class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})

    def laplace_equation(input_, output_):
        force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                      torch.sin(input_.extract(['y'])*torch.pi))
        nabla_u = nabla(output_, input_, components=['u'], d=['x', 'y'])
        return nabla_u - force_term

    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(
            location=Span({'x': [0, 1], 'y':  1}),
            function=nil_dirichlet),
        'gamma2': Condition(
            location=Span({'x': [0, 1], 'y': 0}),
            function=nil_dirichlet),
        'gamma3': Condition(
            location=Span({'x':  1, 'y': [0, 1]}),
            function=nil_dirichlet),
        'gamma4': Condition(
            location=Span({'x': 0, 'y': [0, 1]}),
            function=nil_dirichlet),
        'D': Condition(
            location=Span({'x': [0, 1], 'y': [0, 1]}),
            function=laplace_equation),
        'data': Condition(
            input_points=in_,
            output_points=out_)
    }

    def poisson_sol(self, pts):
        return -(
            torch.sin(pts.extract(['x'])*torch.pi) *
            torch.sin(pts.extract(['y'])*torch.pi)
        )/(2*torch.pi**2)

    truth_solution = poisson_sol


problem = Poisson()

model = FeedForward(problem.input_variables, problem.output_variables)


def test_constructor():
    PINN(problem, model)


def test_span_pts():
    pinn = PINN(problem, model)
    n = 10
    boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    pinn.span_pts(n, 'grid', locations=boundaries)
    for b in boundaries:
        assert pinn.input_pts[b].shape[0] == n
    pinn.span_pts(n, 'random', locations=boundaries)
    for b in boundaries:
        assert pinn.input_pts[b].shape[0] == n

    pinn.span_pts(n, 'grid', locations=['D'])
    assert pinn.input_pts['D'].shape[0] == n**2
    pinn.span_pts(n, 'random', locations=['D'])
    assert pinn.input_pts['D'].shape[0] == n

    pinn.span_pts(n, 'latin', locations=['D'])
    assert pinn.input_pts['D'].shape[0] == n

    pinn.span_pts(n, 'lh', locations=['D'])
    assert pinn.input_pts['D'].shape[0] == n


def test_sampling_all_args():
    pinn = PINN(problem, model)
    n = 10
    pinn.span_pts(n, 'grid', locations=['D'])


def test_sampling_all_kwargs():
    pinn = PINN(problem, model)
    n = 10
    pinn.span_pts(n=n, mode='latin', locations=['D'])


def test_sampling_dict():
    pinn = PINN(problem, model)
    n = 10
    pinn.span_pts(
        {'variables': ['x', 'y'], 'mode': 'grid', 'n': n}, locations=['D'])


def test_sampling_mixed_args_kwargs():
    pinn = PINN(problem, model)
    n = 10
    with pytest.raises(ValueError):
        pinn.span_pts(n, mode='latin', locations=['D'])


def test_train():
    pinn = PINN(problem, model)
    boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    n = 10
    pinn.span_pts(n, 'grid', locations=boundaries)
    pinn.span_pts(n, 'grid', locations=['D'])
    pinn.train(5)


def test_train_2():
    boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    n = 10
    expected_keys = [[], list(range(0, 50, 3))]
    param = [0, 3]
    for i, truth_key in zip(param, expected_keys):
        pinn = PINN(problem, model)
        pinn.span_pts(n, 'grid', locations=boundaries)
        pinn.span_pts(n, 'grid', locations=['D'])
        pinn.train(50, save_loss=i)
        assert list(pinn.history_loss.keys()) == truth_key


def test_train_with_optimizer_kwargs():
    boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    n = 10
    expected_keys = [[], list(range(0, 50, 3))]
    param = [0, 3]
    for i, truth_key in zip(param, expected_keys):
        pinn = PINN(problem, model, optimizer_kwargs={'lr' : 0.3})
        pinn.span_pts(n, 'grid', locations=boundaries)
        pinn.span_pts(n, 'grid', locations=['D'])
        pinn.train(50, save_loss=i)
        assert list(pinn.history_loss.keys()) == truth_key


def test_train_with_lr_scheduler():
    boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    n = 10
    expected_keys = [[], list(range(0, 50, 3))]
    param = [0, 3]
    for i, truth_key in zip(param, expected_keys):
        pinn = PINN(
            problem,
            model,
            lr_scheduler_type=torch.optim.lr_scheduler.CyclicLR,
            lr_scheduler_kwargs={'base_lr' : 0.1, 'max_lr' : 0.3, 'cycle_momentum': False}
        )
        pinn.span_pts(n, 'grid', locations=boundaries)
        pinn.span_pts(n, 'grid', locations=['D'])
        pinn.train(50, save_loss=i)
        assert list(pinn.history_loss.keys()) == truth_key


# def test_train_batch():
#     pinn = PINN(problem, model, batch_size=6)
#     boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
#     n = 10
#     pinn.span_pts(n, 'grid', locations=boundaries)
#     pinn.span_pts(n, 'grid', locations=['D'])
#     pinn.train(5)


# def test_train_batch_2():
#     boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
#     n = 10
#     expected_keys = [[], list(range(0, 50, 3))]
#     param = [0, 3]
#     for i, truth_key in zip(param, expected_keys):
#         pinn = PINN(problem, model, batch_size=6)
#         pinn.span_pts(n, 'grid', locations=boundaries)
#         pinn.span_pts(n, 'grid', locations=['D'])
#         pinn.train(50, save_loss=i)
#         assert list(pinn.history_loss.keys()) == truth_key


if torch.cuda.is_available():

    # def test_gpu_train():
    #     pinn = PINN(problem, model, batch_size=20, device='cuda')
    #     boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    #     n = 100
    #     pinn.span_pts(n, 'grid', locations=boundaries)
    #     pinn.span_pts(n, 'grid', locations=['D'])
    #     pinn.train(5)

    def test_gpu_train_nobatch():
        pinn = PINN(problem, model, batch_size=None, device='cuda')
        boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
        n = 100
        pinn.span_pts(n, 'grid', locations=boundaries)
        pinn.span_pts(n, 'grid', locations=['D'])
        pinn.train(5)
