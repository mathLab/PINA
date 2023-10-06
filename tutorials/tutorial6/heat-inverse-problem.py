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
y_min_r1 = - width/2
y_max_r1 = width/2
x_min_r1 = - width/2
x_max_r1 = width/2
y_min_r2 = - width/4
y_max_r2 = width/4
x_min_r2 = - width/4
x_max_r2 = width/4
v_traverse = torch.tensor([0.01, 0])

# define material properties
k_r1 = 42
k_r2 = k_r1/2
rho = 8
c_p_r1 = 500
c_p_r2 = c_p_r1*1.1
m_r1 = rho*c_p_r1
m_r2 = rho*c_p_r2
t_amb = 0

# read data
FEMmodel = pickle.load(open("data/FEMmodel", "rb"))
NodalSol = pickle.load(open("data/NodalSol", "rb"))
coords_FEM = FEMmodel['Nodes']
time_FEM = NodalSol['XData'].reshape(-1, 1)
laser_power_FEM = FEMmodel["System"]['F']
output_FEM = NodalSol['YData']

# scaling the data
scaler = MinMaxScaler()
output_FEM = scaler.fit_transform(output_FEM.T).T
FEM_points = coords_FEM.shape[0]
FEM_times = time_FEM.shape[0]

## define input points for PINA
coords = LabelTensor(torch.tensor(coords_FEM, dtype=torch.float32),
        labels=['x', 'y'])
input_points = coords.append(LabelTensor(torch.tensor(time_FEM.reshape(FEM_times, 1),
    dtype=torch.float32), labels=['t']), mode='cross')
input_points_init = coords.append(LabelTensor(torch.tensor(time_FEM[0].reshape(1, 1),
    dtype=torch.float32), labels=['t']), mode='cross')

## define output points for PINA
output_points = output_FEM[:FEM_times, :].flatten().reshape(FEM_times*FEM_points, 1)
output_points = LabelTensor(torch.tensor(output_points, dtype=torch.float32),
    labels=['T'])

# define laser properties and heat source
A_l = (torch.pi*0.002**2)/4
q_L = 807.1429/A_l*100
cov = 0.002/3.0349
sigma = np.array([cov, cov])
mean = np.array([-width/4, width/4])
source_calculated = np.exp(
        -((coords_FEM[:, 0] - mean[0])**2)/(2*(sigma[0]**2))
        -((coords_FEM[:, 1] - mean[1])**2)/(2*(sigma[1]**2))
        )
#plt.tricontourf(triangulation, output_FEM[-1].flatten())
#plt.colorbar()
#plt.show()

## define domain properties for PINA
t_min = time_FEM[0]
t_max = time_FEM[-1]
n_steps = FEM_times
n_points = 10
n_points_boundaries = 100
n_steps_boundaries = 100

class ThermalProblem(SpatialProblem, TimeDependentProblem, InverseProblem):
    '''
    Thermal inverse problem class for PINA.
    '''
    output_variables = ['T']
    spatial_domain = CartesianDomain({'x': [x_min_r1, x_max_r1],
        'y': [y_min_r1, y_max_r1]})
    temporal_domain = CartesianDomain({'t':[t_min, t_max]})

    inferred_parameters = torch.nn.Parameter(torch.rand(3, requires_grad=True))
    inferred_domain = CartesianDomain({'m':[0, 10], 'k':[0, 10], 'c':[0, 10]})

    def thermal_equation_internal(input_, output_, inferred_parameters):
        # time derivative
        enthalpy_change = inferred_parameters[0]*grad(output_, input_,
                components=['T'], d=['t'])
        # laplacian
        heat_conduction = inferred_parameters[1]*laplacian(
                output_, input_, components=['T'],
                d=['x', 'y'])
        # heat source 1: laser power
        center_source = torch.tensor([-width/4, width/4])
        center_source = center_source + v_traverse*input_.extract(['t'])
        heat_source_laser = (
        -((input_.extract(['x']) - center_source[0])**2)/(2*(sigma[0]**2))
        -((input_.extract(['y']) - center_source[1])**2)/(2*(sigma[1]**2))
        ).exp()
        # heat source 2: convection
        heat_source_conv = - inferred_parameters[2]*(
                output_.extract(['T']) - t_amb)
        # heat source 3: radiation
#        heat_source_rad = - const_rad*(output_.extract(['T'])**4 - t_amb**4)
        heat_source_rad = 0

        return (enthalpy_change - heat_conduction - (heat_source_conv
             + heat_source_laser + heat_source_rad))

    # problem condition statement
    conditions = {
        'gamma_top': Condition(location=CartesianDomain({
            'x': [x_min_r1, x_max_r1], 'y':  y_max_r1, 't':[t_min, t_max]}),
            equation=FixedValue(t_amb, components=['T'])),
        'gamma_bot': Condition(location=CartesianDomain({
            'x': [x_min_r1, x_max_r1], 'y': y_min_r1, 't':[t_min, t_max]}),
            equation=FixedValue(t_amb, components=['T'])),
        'gamma_out': Condition(location=CartesianDomain({
            'x':  x_max_r1, 'y': [y_min_r1, y_max_r1], 't':[t_min, t_max]}),
            equation=FixedValue(t_amb, components=['T'])),
        'gamma_in':  Condition(location=CartesianDomain({
            'x': x_min_r1, 'y': [y_min_r1, y_max_r1], 't':[t_min, t_max]}),
            equation=FixedValue(t_amb, components=['T'])),
        'data': Condition(input_points=input_points,
            output_points=output_points),
        'D_int': Condition(location=CartesianDomain({
            'x': [x_min_r2, x_max_r2], 'y': [y_min_r2, y_max_r2], 't':[t_min, t_max]}),
            equation=ParametricEquation(thermal_equation_internal)),
        'init': Condition(location=CartesianDomain({
            'x': [x_min_r1, x_max_r1], 'y': [y_min_r1, y_max_r1], 't': 0}),
            equation=FixedValue(t_amb, components=['T'])),
        }

class Feature(torch.nn.Module):
    '''
    Extra-feature of source power of laser.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        center_source = torch.tensor([-width/4, width/4])
        center_source = center_source + v_traverse*x.extract(['t'])
        heat_source_laser = (
        -((x.extract(['x']) - center_source[0])**2)/(2*(sigma[0]**2))
        -((x.extract(['y']) - center_source[1])**2)/(2*(sigma[1]**2))
        ).exp()
        return LabelTensor(heat_source_laser, ['feat'])

class GaussianLayer(torch.nn.Module):
    """
    Define a network with only one layer as a composition of gaussian functions.
    The PINN solution will be the sum of all the gaussians.
    """
    def __init__(self, n_gaussians):
        super().__init__()
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
        # define the gaussians (evolved in time)
        for i in range(self.n_gaussians):
            gaussians.append(torch.exp(
                -(x_coord - x1_evolved[:, i])**2/(2*(self.alpha1[i]**2))
                -(y_coord - self.x2[i])**2/(2*(self.alpha2[i]**2))))
        # set to zero the weights of the gaussians that are too small
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

    tmp_dir = 'tmp_heat_inverse'

    epochs = 10000

    # decide the number of gaussians for approximating the field
    n_gaussians = 20

    model = GaussianLayer(n_gaussians)

    pinn = PINN(
        problem=thermal_problem,
        model=model,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.0003, 'weight_decay': 1e-9},
        )

    thermal_problem.discretise_domain(
        n=n_steps_boundaries, mode='grid', variables=['t'],
        locations=['gamma_in', 'gamma_out', 'gamma_top', 'gamma_bot']
    )
    thermal_problem.discretise_domain(
        n=n_points_boundaries, mode='grid', variables=['x', 'y'],
        locations=['gamma_in', 'gamma_out', 'gamma_top', 'gamma_bot']
    )
    thermal_problem.discretise_domain(
        n=n_steps, mode='grid', variables=['t'],
        locations=['D_int']
    )
    thermal_problem.discretise_domain(
        n=n_points_boundaries, mode='grid', variables=['x', 'y'],
        locations=['D_int']
    )
    thermal_problem.discretise_domain(
        n=10, mode='grid',
        locations=['init']
        )

    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=epochs,
                default_root_dir=tmp_dir)

    if args.s:
        trainer.train()
        plotter = Plotter()
        plotter.plot(pinn, components='T', fixed_variables={'t': 0})
        plotter.plot(pinn, components='T', fixed_variables={'t': 0.3})

    if args.l:
        pinn = PINN.load_from_checkpoint(
            '{}/lightning_logs/version_0/checkpoints/epoch={}-step={}.ckpt'.format(
                tmp_dir,folder_checkpoints, epochs-1, epochs),
            problem=thermal_problem, model=model)

        plotter = Plotter()
        plotter.plot(pinn, components='T', fixed_variables={'t': time_FEM[0]})
        plotter.plot(pinn, components='T', fixed_variables={'t': time_FEM[1]})

