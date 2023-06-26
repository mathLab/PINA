import torch
import math
import faiss

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


def symsqrt(matrix):
    _, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def scale(C, x):
    return 1 / math.sqrt(2) * torch.matmul(C, x.T).mT.double()


C = torch.zeros(3, 3, dtype=torch.float32)
C[0, 0], C[1, 1], C[2, 2] = 1 - nu, 1 - nu, (1 - 2 * nu)
C[0, 1], C[1, 0] = nu, nu
C = E / ((1 + nu) * (1 - 2 * nu)) * C
Cinv = torch.inverse(C)
Csqrt, Cisqrt = symsqrt(C), symsqrt(torch.inverse(C))


# Define data
def create_data(ndata):
    data_eps = torch.distributions.Uniform(-0.1, 0.1).sample((ndata, 3))
    data_sig = torch.matmul(C, data_eps.T).mT
    data_eps_sc = scale(Csqrt, data_eps).float()
    data_sig_sc = scale(Cisqrt, data_sig).float()
    data_sc = torch.stack((data_eps_sc, data_sig_sc), dim=1).reshape(ndata, 6)
    return data_eps, data_sig, data_sc


data_eps, data_sig, data_sc = create_data(5000000)
nlist = 1000
k = 6
quantizer = faiss.IndexFlatL2(6)  # the other index
index = faiss.IndexIVFFlat(quantizer, 6, nlist)
assert not index.is_trained
index.train(data_sc.detach().numpy())
assert index.is_trained
index.add(data_sc.detach().numpy())
index.nprobe = 100


def equilibrium(input_, output_):
    output_grad = grad(output_, input_)
    Gex = output_grad.extract(['ds11dx']) + output_grad.extract(['ds12dy'])
    Gey = output_grad.extract(['ds12dx']) + output_grad.extract(['ds22dy'])
    return torch.stack([Gex, Gey], dim=1).squeeze()


def distance(input_, output_):
    output_grad = grad(output_, input_)
    Gex = output_grad.extract(['ds11dx']) + output_grad.extract(['ds12dy'])
    Gey = output_grad.extract(['ds12dx']) + output_grad.extract(['ds22dy'])
    e11 = output_grad.extract(['du1dx'])
    e22 = output_grad.extract(['du2dy'])
    e12 = 0.5 * (output_grad.extract(['du1dy']) + output_grad.extract(['du2dx']))
    s11 = output_.extract(['s11'])
    s22 = output_.extract(['s22'])
    s12 = output_.extract(['s12'])

    eps = torch.stack((e11, e22, e12), dim=1).detach().clone().squeeze()
    sig = torch.stack((s11, s22, s12), dim=1).detach().clone().squeeze()
    points = torch.stack((scale(Csqrt, eps), scale(Cisqrt, sig)), dim=1).reshape(len(e11), 6)
    _, indx = index.search(points.detach().numpy(), 1)

    eps_opt = data_eps[indx.squeeze()]
    sig_opt = data_sig[indx.squeeze()]
    e11_opt = eps_opt[:, 0][:, None]
    e22_opt = eps_opt[:, 1][:, None]
    e12_opt = eps_opt[:, 2][:, None]
    s11_opt = sig_opt[:, 0][:, None]
    s22_opt = sig_opt[:, 1][:, None]
    s12_opt = sig_opt[:, 2][:, None]
    return torch.stack(
        [Gex, Gey, e11 - e11_opt, e22 - e22_opt, e12 - e12_opt, s11 - s11_opt, s22 - s22_opt, s12 - s12_opt],
        dim=1).squeeze()


class Mechanics(SpatialProblem):
    output_variables = ['u1', 'u2', 's11', 's22', 's12']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'D': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': [0, 1]}),
            equation=SystemEquation([distance]))
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
                                          torch.nn.Linear(20, 20),
                                          torch.nn.Tanh(),
                                          torch.nn.Linear(20, output_dim))

    # here in the foward we implement the hard constraints
    def forward(self, x):
        output = self.layers(x)
        delta = 0.05
        u1_hard = delta * x.extract(['x']) + (1 - x.extract(['x'])) * x.extract(['x']) * output[:, 0][:, None]
        u2_hard = x.extract(['y']) * output[:, 1][:, None]
        modified_output = torch.hstack(
            [u1_hard, u2_hard, output[:, 2][:, None], output[:, 3][:, None], output[:, 4][:, None]])
        return modified_output


model = HardMLP(len(bvp_problem.input_variables), len(bvp_problem.output_variables))
bvp_problem.discretise_domain(50, 'grid', locations=['D'])

# make the solver
solver = PINN(problem=bvp_problem, model=model, optimizer=torch.optim.LBFGS)

# train the model (ONLY CPU for now, all other devises in the official release)
trainer = Trainer(solver=solver, kwargs={'max_epochs': 5000, 'accelerator': 'cpu', 'deterministic': True})
trainer.train()

# plotter
plotter = Plotter()
plotter.plot(solver=solver, components='u1')
plotter.plot(solver=solver, components='u2')

# get components ui on pts
v = [var for var in solver.problem.input_variables]
pts = solver.problem.domain.sample(256, 'grid', variables=v)
predicted_output = solver.forward(pts)
u1 = predicted_output.extract('u1')
u2 = predicted_output.extract('u2')

import matplotlib.pyplot as plt

cmap = 'jet'
plt.figure()
plt.scatter(pts.detach().numpy()[:, 0], pts.detach().numpy()[:, 1], s=5, c=u1.detach().numpy(), cmap=cmap)
plt.colorbar()
plt.savefig("C:/Users/Kerem/PycharmProjects/PINA/tutorials/tutorial 5/results/u1")
plt.figure()
plt.scatter(pts.detach().numpy()[:, 0], pts.detach().numpy()[:, 1], s=5, c=u2.detach().numpy(), cmap=cmap)
plt.colorbar()
plt.savefig("C:/Users/Kerem/PycharmProjects/PINA/tutorials/tutorial 5/results/u2")
