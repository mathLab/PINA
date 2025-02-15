import torch
from pina.problem import SpatialProblem, InverseProblem
from pina.operators import laplacian
from pina.domain import CartesianDomain
from pina import Condition, LabelTensor
from pina.solvers import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.equation import Equation
from pina.equation.equation_factory import FixedValue
from pina.loss import LpLoss
from pina.problem.zoo import Poisson2DSquareProblem

# class InversePoisson(SpatialProblem, InverseProblem):
#     '''
#     Problem definition for the Poisson equation.
#     '''
#     output_variables = ['u']
#     x_min = -2
#     x_max = 2
#     y_min = -2
#     y_max = 2
#     data_input = LabelTensor(torch.rand(10, 2), ['x', 'y'])
#     data_output = LabelTensor(torch.rand(10, 1), ['u'])
#     spatial_domain = CartesianDomain({'x': [x_min, x_max], 'y': [y_min, y_max]})
#     # define the ranges for the parameters
#     unknown_parameter_domain = CartesianDomain({'mu1': [-1, 1], 'mu2': [-1, 1]})

#     def laplace_equation(input_, output_, params_):
#         '''
#         Laplace equation with a force term.
#         '''
#         force_term = torch.exp(
#                 - 2*(input_.extract(['x']) - params_['mu1'])**2
#                 - 2*(input_.extract(['y']) - params_['mu2'])**2)
#         delta_u = laplacian(output_, input_, components=['u'], d=['x', 'y'])

#         return delta_u - force_term

#     # define the conditions for the loss (boundary conditions, equation, data)
#     conditions = {
#         'gamma1': Condition(domain=CartesianDomain({'x': [x_min, x_max],
#             'y':  y_max}),
#             equation=FixedValue(0.0, components=['u'])),
#         'gamma2': Condition(domain=CartesianDomain(
#             {'x': [x_min, x_max], 'y': y_min
#             }),
#             equation=FixedValue(0.0, components=['u'])),
#         'gamma3': Condition(domain=CartesianDomain(
#             {'x':  x_max, 'y': [y_min, y_max]
#             }),
#             equation=FixedValue(0.0, components=['u'])),
#         'gamma4': Condition(domain=CartesianDomain(
#             {'x': x_min, 'y': [y_min, y_max]
#             }),
#             equation=FixedValue(0.0, components=['u'])),
#         'D': Condition(domain=CartesianDomain(
#             {'x': [x_min, x_max], 'y': [y_min, y_max]
#             }),
#         equation=Equation(laplace_equation)),
#         'data': Condition(input_points=data_input.extract(['x', 'y']),
#                           output_points=data_output)
#     }


# # make the problem
# poisson_problem = Poisson2DSquareProblem()
# model = FeedForward(len(poisson_problem.input_variables),
#                     len(poisson_problem.output_variables))
# model_extra_feats = FeedForward(
#     len(poisson_problem.input_variables) + 1,
#     len(poisson_problem.output_variables))


# def test_constructor():
#     PINN(problem=poisson_problem, model=model, extra_features=None)


# def test_constructor_extra_feats():
#     model_extra_feats = FeedForward(
#         len(poisson_problem.input_variables) + 1,
#         len(poisson_problem.output_variables))
#     PINN(problem=poisson_problem,
#          model=model_extra_feats)


# def test_train_cpu():
#     poisson_problem = Poisson2DSquareProblem()
#     boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
#     n = 10
#     poisson_problem.discretise_domain(n, 'grid', locations=boundaries)
#     pinn = PINN(problem = poisson_problem, model=model,
#                 extra_features=None, loss=LpLoss())
#     trainer = Trainer(solver=pinn, max_epochs=1,
#                       accelerator='cpu', batch_size=20, val_size=0., train_size=1., test_size=0.)

# def test_train_load():
#     tmpdir = "tests/tmp_load"
#     poisson_problem = Poisson2DSquareProblem()
#     boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
#     n = 10
#     poisson_problem.discretise_domain(n, 'grid', locations=boundaries)
#     pinn = PINN(problem=poisson_problem,
#                 model=model,
#                 extra_features=None,
#                 loss=LpLoss())
#     trainer = Trainer(solver=pinn,
#                       max_epochs=15,
#                       accelerator='cpu',
#                       default_root_dir=tmpdir)
#     trainer.train()
#     new_pinn = PINN.load_from_checkpoint(
#         f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=14-step=15.ckpt',
#         problem = poisson_problem, model=model)
#     test_pts = CartesianDomain({'x': [0, 1], 'y': [0, 1]}).sample(10)
#     assert new_pinn.forward(test_pts).extract(['u']).shape == (10, 1)
#     assert new_pinn.forward(test_pts).extract(
#         ['u']).shape == pinn.forward(test_pts).extract(['u']).shape
#     torch.testing.assert_close(
#         new_pinn.forward(test_pts).extract(['u']),
#         pinn.forward(test_pts).extract(['u']))
#     import shutil
#     shutil.rmtree(tmpdir)

# def test_train_restore():
#     tmpdir = "tests/tmp_restore"
#     poisson_problem = Poisson2DSquareProblem()
#     boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
#     n = 10
#     poisson_problem.discretise_domain(n, 'grid', locations=boundaries)
#     pinn = PINN(problem=poisson_problem,
#                 model=model,
#                 extra_features=None,
#                 loss=LpLoss())
#     trainer = Trainer(solver=pinn,
#                       max_epochs=5,
#                       accelerator='cpu',
#                       default_root_dir=tmpdir)
#     trainer.train()
#     ntrainer = Trainer(solver=pinn, max_epochs=15, accelerator='cpu')
#     t = ntrainer.train(
#         ckpt_path=f'{tmpdir}/lightning_logs/version_0/'
#                   'checkpoints/epoch=4-step=5.ckpt')
#     import shutil
#     shutil.rmtree(tmpdir)

# def test_train_inverse_problem_cpu():
#     poisson_problem = InversePoisson()
#     boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4', 'D']
#     n = 100
#     poisson_problem.discretise_domain(n, 'random', locations=boundaries,
#                                       variables=['x', 'y'])
#     pinn = PINN(problem = poisson_problem, model=model,
#                 extra_features=None, loss=LpLoss())
#     trainer = Trainer(solver=pinn, max_epochs=1,
#                       accelerator='cpu', batch_size=20)
#     trainer.train()

# def test_train_inverse_problem_load():
#     tmpdir = "tests/tmp_load_inv"
#     poisson_problem = InversePoisson()
#     boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4', 'D']
#     n = 100
#     poisson_problem.discretise_domain(n, 'random', locations=boundaries)
#     pinn = PINN(problem=poisson_problem,
#                 model=model,
#                 extra_features=None,
#                 loss=LpLoss())
#     trainer = Trainer(solver=pinn,
#                       max_epochs=15,
#                       accelerator='cpu',
#                       default_root_dir=tmpdir)
#     trainer.train()
#     new_pinn = PINN.load_from_checkpoint(
#         f'{tmpdir}/lightning_logs/version_0/checkpoints/epoch=14-step=15.ckpt',
#         problem = poisson_problem, model=model)
#     test_pts = CartesianDomain({'x': [0, 1], 'y': [0, 1]}).sample(10)
#     assert new_pinn.forward(test_pts).extract(['u']).shape == (10, 1)
#     assert new_pinn.forward(test_pts).extract(
#         ['u']).shape == pinn.forward(test_pts).extract(['u']).shape
#     torch.testing.assert_close(
#         new_pinn.forward(test_pts).extract(['u']),
#         pinn.forward(test_pts).extract(['u']))
#     import shutil
#     shutil.rmtree(tmpdir)