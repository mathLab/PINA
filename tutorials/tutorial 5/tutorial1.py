import torch
from pina.problem import SpatialProblem
from pina.operators import grad
from pina.geometry import CartesianDomain
from pina import Condition, PINN
from pina.trainer import Trainer
from pina.equation.system_equation import SystemEquation
from pina.plotter import Plotter

# Define material
E = 7
nu = 0.3
p = 'plain_strain'

if p == 'plain_strain':  ### plain strain
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / (1 + nu) / 2
elif p == 'plain_stress':  ### plain stress
    lmbda = E * nu / (1 + nu) / (1 - nu)
    mu = E / (1 + nu) / 2

def material(input_, output_):
    u_grad = grad(output_, input_)
    u1_xx = grad(u_grad, input_, components=['du1dx'], d=['x'])
    u1_xy = grad(u_grad, input_, components=['du1dx'], d=['y'])
    u1_yx = grad(u_grad, input_, components=['du1dy'], d=['x'])
    u1_yy = grad(u_grad, input_, components=['du1dy'], d=['y'])

    u2_xx = grad(u_grad, input_, components=['du2dx'], d=['x'])
    u2_xy = grad(u_grad, input_, components=['du2dx'], d=['y'])
    u2_yx = grad(u_grad, input_, components=['du2dy'], d=['x'])
    u2_yy = grad(u_grad, input_, components=['du2dy'], d=['y'])

    ### Calculate strain
    e11 = u_grad.extract(['du1dx'])
    e22 = u_grad.extract(['du2dy'])
    e12 = 0.5 * (u_grad.extract(['du1dy']) + u_grad.extract(['du2dx']))

    ### Calculate stress
    s11 = (2 * mu + lmbda) * e11 + lmbda * e22
    s22 = (2 * mu + lmbda) * e22 + lmbda * e11
    s12 = 2 * mu * e12

    ### Calculate equilibrium
    Gex = (2 * mu + lmbda) * u1_xx + mu * u1_yy + mu * u2_xy + lmbda * u2_yx
    Gey = (2 * mu + lmbda) * u2_yy + mu * u2_xx + mu * u1_yx + lmbda * u1_xy

    return e11, e22, e12, s11, s22, s12, Gex, Gey


def equilibrium(input_, output_):
    _, _, _, _, _, _, Gex, Gey = material(input_, output_)
    return torch.stack([Gex, Gey], dim=1)


class Mechanics(SpatialProblem):
    output_variables = ['u1', 'u2']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'D': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': [0, 1]}),
            equation=SystemEquation([equilibrium]))
    }

# make the problem
bvp_problem = Mechanics()

class HardMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, 20),
                                          torch.nn.Tanh(),
                                          torch.nn.Linear(20, 20),
                                          torch.nn.Tanh(),
                                          torch.nn.Linear(20, output_dim))

    def forward(self, x):
        output = self.layers(x)
        delta = 0.05
        u1_hard = delta * x.extract(['x']) + (1 - x.extract(['x'])) * x.extract(['x']) * output[:, 0][:, None]
        u2_hard = x.extract(['y']) * output[:, 1][:, None]
        modified_output = torch.hstack([u1_hard, u2_hard])
        return modified_output

model = HardMLP(len(bvp_problem.input_variables), len(bvp_problem.output_variables))
bvp_problem.discretise_domain(50, 'grid', locations=['D'])

# make the solver
solver = PINN(problem=bvp_problem, model=model, optimizer=torch.optim.LBFGS)

# train the model (ONLY CPU for now, all other devises in the official release)
#logger = pl_loggers.TensorBoardLogger(name=f'ssl-{0.01}-{1}', save_dir='runs')

trainer = Trainer(solver=solver, kwargs={'max_epochs': 1, 'accelerator': 'cpu', 'deterministic': True})
trainer.train()

# plotter
plotter = Plotter()
plotter.plot(solver=solver, components='u1')
plotter.plot(solver=solver, components='u2')

# get components ui on pts
v = [var for var in solver.problem.input_variables]
pts = solver.problem.domain.sample(256, 'grid', variables=v)
pts.requires_grad = True
predicted_output = solver.forward(pts)
u1 = predicted_output.extract('u1')
u2 = predicted_output.extract('u2')

e11, e22, e12, s11, s22, s12, _, _ = material(pts, solver.forward(pts))

import matplotlib.pyplot as plt
cmap = 'jet'
plt.figure()
plt.scatter(pts.detach().numpy()[:, 0], pts.detach().numpy()[:, 1], s=5, c=u1.detach().numpy(), cmap=cmap)
plt.colorbar()
plt.savefig("results/u1")
plt.figure()
plt.scatter(pts.detach().numpy()[:, 0], pts.detach().numpy()[:, 1], s=5, c=u2.detach().numpy(), cmap=cmap)
plt.colorbar()
plt.savefig("results/u2")
plt.figure()
plt.plot(solver.loss_list)
plt.savefig("results/loss")