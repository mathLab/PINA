import torch

from pina.problem import AbstractProblem
from pina import Condition, LabelTensor
from pina.solvers import GAROM
from pina.trainer import Trainer
import torch.nn as nn
import matplotlib.tri as tri


def func(x, mu1, mu2):
    import torch
    x_m1 = (x[:, 0] - mu1).pow(2)
    x_m2 = (x[:, 1] - mu2).pow(2)
    norm = x[:, 0]**2 + x[:, 1]**2
    return torch.exp(-(x_m1 + x_m2))

class ParametricGaussian(AbstractProblem):
    output_variables = [f'u_{i}' for i in range(900)]

    # params
    xx = torch.linspace(-1, 1, 20)
    yy = xx
    params = LabelTensor(torch.cartesian_prod(xx, yy), labels=['mu1', 'mu2'])

    # define domain
    x =  torch.linspace(-1, 1, 30)
    domain = torch.cartesian_prod(x, x)
    triang = tri.Triangulation(domain[:, 0], domain[:, 1])
    sol = []
    for p in params:
        sol.append(func(domain, p[0], p[1]))
    snapshots = LabelTensor(torch.stack(sol), labels=output_variables)

    # define conditions
    conditions = {
        'data': Condition(
            input_points=params,
            output_points=snapshots)
    }

# simple Generator Network
class Generator(nn.Module):
    def __init__(self, input_dimension, parameters_dimension,
                 noise_dimension, activation=torch.nn.SiLU):
        super().__init__()

        self._noise_dimension = noise_dimension
        self._activation = activation

        self.model = torch.nn.Sequential(
            torch.nn.Linear(6 * self._noise_dimension, input_dimension // 6),
            self._activation(),
            torch.nn.Linear(input_dimension // 6, input_dimension // 3),
            self._activation(),
            torch.nn.Linear(input_dimension // 3, input_dimension)
        )        
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(parameters_dimension, 2 * self._noise_dimension),
            self._activation(),
            torch.nn.Linear(2 * self._noise_dimension, 5 * self._noise_dimension)
        )

    def forward(self, param):
        # uniform sampling in [-1, 1]
        z = torch.rand(size=(param.shape[0], self._noise_dimension),
                       device=param.device,
                       dtype=param.dtype,
                       requires_grad=True)
        z = 2. * z - 1.

        # conditioning by concatenation of mapped parameters
        input_ = torch.cat((z, self.condition(param)), dim=-1)
        out = self.model(input_)

        return out


# Simple Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dimension, parameter_dimension,
                 hidden_dimension, activation=torch.nn.ReLU):
        super().__init__()

        self._activation = activation
        self.encoding = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, input_dimension // 3),
            self._activation(),
            torch.nn.Linear(input_dimension // 3, input_dimension // 6),
            self._activation(),
            torch.nn.Linear(input_dimension // 6, hidden_dimension)
        )
        self.decoding = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dimension, input_dimension // 6),
            self._activation(),
            torch.nn.Linear(input_dimension // 6, input_dimension // 3),
            self._activation(),
            torch.nn.Linear(input_dimension // 3, input_dimension),
        )

        self.condition = torch.nn.Sequential(
            torch.nn.Linear(parameter_dimension, hidden_dimension // 2),
            self._activation(),
            torch.nn.Linear(hidden_dimension // 2, hidden_dimension)
        )
       
    def forward(self, data):
        x, condition = data
        encoding = self.encoding(x)
        conditioning = torch.cat((encoding, self.condition(condition)), dim=-1)
        decoding = self.decoding(conditioning)
        return decoding


problem = ParametricGaussian()

def test_constructor():
    GAROM(problem = problem,
          generator = Generator(input_dimension=900,
                                parameters_dimension=2,
                                noise_dimension=12),
          discriminator = Discriminator(input_dimension=900,
                                        parameter_dimension=2,
                                        hidden_dimension=64)
          )

def test_train_cpu():
    solver = GAROM(problem = problem,
                generator = Generator(input_dimension=900,
                                        parameters_dimension=2,
                                        noise_dimension=12),
                discriminator = Discriminator(input_dimension=900,
                                                parameter_dimension=2,
                                                hidden_dimension=64)
                )

    trainer = Trainer(solver=solver, max_epochs=4, accelerator='cpu')
    trainer.train()

def test_sample():
    solver = GAROM(problem = problem,
                generator = Generator(input_dimension=900,
                                        parameters_dimension=2,
                                        noise_dimension=12),
                discriminator = Discriminator(input_dimension=900,
                                                parameter_dimension=2,
                                                hidden_dimension=64)
                )
    solver.sample(problem.params)
    assert solver.sample(problem.params).shape == problem.snapshots.shape

def test_forward():
    solver = GAROM(problem = problem,
                generator = Generator(input_dimension=900,
                                        parameters_dimension=2,
                                        noise_dimension=12),
                discriminator = Discriminator(input_dimension=900,
                                                parameter_dimension=2,
                                                hidden_dimension=64)
                )
    solver(problem.params, mc_steps=100, variance=True)
    assert solver(problem.params).shape == problem.snapshots.shape