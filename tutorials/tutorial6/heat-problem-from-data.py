import numpy as np
from pina.problem import SpatialProblem, TimeDependentProblem, InverseProblem
from pina.operators import laplacian, grad, div, advection
from pina import Condition, LabelTensor, CartesianDomain, Trainer
from pina.equation import (Equation, SystemEquation, FixedGradient, FixedValue,
ParametricEquation)
from pina.geometry import Exclusion
import torch
from matplotlib import pyplot as plt
import argparse
import time
from pina import PINN, LabelTensor, Plotter
from pina.model import FeedForward, DeepONet
import pickle
from torch.nn import Softplus
from matplotlib import tri
from sklearn.preprocessing import MinMaxScaler

# define spatial domain ranges
width = 0.05
y_min = - width/2
y_max = width/2
x_min = - width/2
x_max = width/2

# define laser properties
v_traverse = torch.tensor([0.01, 0])
cov = 0.002/3.0349
sigma = np.array([cov, cov])

# time indeces for FEM data (we consider last 100 time steps)
start_time = -100
end_time = -1

# read FEM data
FEMmodel = pickle.load(open("data/FEMmodel", "rb"))
NodalSol = pickle.load(open("data/NodalSol", "rb"))
coords_FEM = FEMmodel['Nodes']
time_FEM = NodalSol['XData'][start_time:end_time].reshape(-1, 1)
output_FEM = NodalSol['YData'][start_time:end_time]
FEM_points = coords_FEM.shape[0]
FEM_times = time_FEM.shape[0]

# rescale FEM data: it makes the training easier
scaler = MinMaxScaler()
output_FEM = scaler.fit_transform(output_FEM.T).T
#triangulation = tri.Triangulation(coords_FEM[:, 0], coords_FEM[:, 1])
#plt.tricontourf(triangulation, output_FEM[0].flatten())
#plt.colorbar()
#plt.show()

## define input points for PINA (as LabelTensor)
coords = LabelTensor(torch.tensor(coords_FEM, dtype=torch.float32),
        labels=['x', 'y'])
input_points = coords.append(LabelTensor(torch.tensor(time_FEM.reshape(FEM_times, 1),
    dtype=torch.float32), labels=['t']), mode='cross')
input_points_init = coords.append(LabelTensor(torch.tensor(time_FEM[0].reshape(1, 1),
    dtype=torch.float32), labels=['t']), mode='cross')

# define laser properties
## define output points for PINA (as LabelTensor)
output_points = output_FEM[:FEM_times, :].flatten().reshape(FEM_times*FEM_points, 1)
output_points = LabelTensor(torch.tensor(output_points, dtype=torch.float32),
    labels=['T'])

## define domain properties for PINA
t_min = time_FEM[0]
t_max = time_FEM[-1]
n_steps = FEM_times

class ThermalProblem(SpatialProblem, TimeDependentProblem):

    # assign output/ spatial variables
    output_variables = ['T']
    spatial_domain = CartesianDomain({'x': [x_min, x_max],
        'y': [y_min, y_max]})
    temporal_domain = CartesianDomain({'t':[t_min, t_max]})

    # problem condition statement
    conditions = {
        'data': Condition(input_points=input_points,
            output_points=output_points),
        }

class GaussianLayer(torch.nn.Module):
    """
    Define a network with only one layer as a composition of gaussian functions.
    The PINN solution will be the sum of all the gaussians.
    """
    def __init__(self, n_gaussians):
        super().__init__()
        # define the gaussians parameters as torch parameters
        self.n_gaussians = n_gaussians
        self.x1 = torch.nn.Parameter(torch.rand(self.n_gaussians)*(width/2) - (width/4))
        self.x2 = torch.nn.Parameter(torch.rand(self.n_gaussians)*(width/2) - (width/4))
        self.alpha1 = torch.nn.Parameter(torch.rand(self.n_gaussians)*cov)
        self.alpha2 = torch.nn.Parameter(torch.rand(self.n_gaussians)*cov)
        self.weights = torch.nn.Parameter(torch.rand(self.n_gaussians))
        self.velocities = torch.nn.Parameter(torch.rand(self.n_gaussians)*0.01)

    def forward(self, x):
        x_coord = x.extract(['x'])
        y_coord = x.extract(['y'])
        t = x.extract(['t'])
        gaussians = []
        x1_evolved = self.x1 + self.velocities*t
        # define all the gaussians and the solution as a linear combination
        for i in range(self.n_gaussians):
            gaussians.append(torch.exp(
                -(x_coord - x1_evolved[:, i])**2/(2*(self.alpha1[i]**2))
                -(y_coord - self.x2[i])**2/(2*(self.alpha2[i]**2))))
            # if the gaussian is too small, we set the weight to zero
            if self.alpha1[i] < 1e-6 or self.alpha2[i] < 1e-6:
                self.weights.data[i] = 0.
        gaussians = [(self.weights[i]**2)*gaussians[i]
                for i in range(self.n_gaussians)]
        # linear combination of gaussians with weights
        gaussians = torch.stack(gaussians)
        out = torch.sum(gaussians, dim=0)
        return LabelTensor(out, labels=['T'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    args = parser.parse_args()

    thermal_problem = ThermalProblem()

    # temporary directory for saving/loading model
    tmp_dir = 'tmp_heat_data'

    # number of epochs for training
    epochs = 10000

    # decide the number of gaussians for approximating the field
    n_gaussians = 20

    model = GaussianLayer(n_gaussians)

    # PINN definition
    pinn = PINN(
        problem=thermal_problem,
        model=model,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.0003, 'weight_decay': 1e-9},
        )

    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=epochs,
                default_root_dir=tmp_dir)

    if args.s:
        # training the model
        trainer.train()
        plotter = Plotter()
        plotter.plot(pinn, components='T', fixed_variables={'t': time_FEM[0]})
        plotter.plot(pinn, components='T', fixed_variables={'t': time_FEM[-1]})

    if args.l:
        pinn = PINN.load_from_checkpoint(
            '{}/lightning_logs/version_0/checkpoints/epoch={}-step={}.ckpt'.format(
                tmp_dir, epochs-1, epochs),
            problem=thermal_problem, model=model)

        plotter = Plotter()

        sol_to_plot = pinn(input_points[:coords.size()[0], :]).detach().numpy().flatten()
        triangulation = tri.Triangulation(coords_FEM[:, 0], coords_FEM[:, 1])

        # plot the PINN solution, the FEM solution and the absolute error
        fig, axs = plt.subplots(1, 3, figsize=(30, 5))
        c_PINN = axs[0].tricontourf(triangulation, sol_to_plot)
        axs[0].set_title('PINN solution')
        fig.colorbar(c_PINN, ax=axs[0])
        c_FEM = axs[1].tricontourf(triangulation, output_FEM[0].flatten())
        axs[1].set_title('FEM solution')
        fig.colorbar(c_FEM, ax=axs[1])
        c_ERR = axs[2].tricontourf(triangulation, np.abs(sol_to_plot - output_FEM[0].flatten()))
        axs[2].set_title('Absolute error')
        fig.colorbar(c_ERR, ax=axs[2])
        plt.show()



