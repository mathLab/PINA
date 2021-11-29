import numpy as np
import torch
from pina.problem import Problem
from pina.segment import Segment
from pina.parametricproblem2d import ParametricProblem2D

bc = {
    'y': (
        (Segment((0, 0), (4, 0)), lambda x: torch.ones(x.shape[0], 1)),
        (Segment((4, 0), (4, 1)), lambda x: torch.ones(x.shape[0], 1)),
        (Segment((4, 1), (0, 1)), lambda x: torch.ones(x.shape[0], 1)),
        (Segment((0, 1), (0, 0)), lambda x: torch.ones(x.shape[0], 1)),
    ),
    'p': (
        (Segment((0, 0), (4, 0)), lambda x: torch.zeros(x.shape[0], 1)),
        (Segment((4, 0), (4, 1)), lambda x: torch.zeros(x.shape[0], 1)),
        (Segment((4, 1), (0, 1)), lambda x: torch.zeros(x.shape[0], 1)),
        (Segment((0, 1), (0, 0)), lambda x: torch.zeros(x.shape[0], 1)),
    )
}

# optimal control parameters and data
alpha = 1e-5
# yd = 10*x[:, 0]*(1-x[:, 0])*x[:, 1]*(1-x[:, 1])
# three variables 
# state y = f[0]
# control u = f[1]
# adjoint p = f[2]

# the three variables

def adjoint_eq(x, f):
    grad_x, grad_y = Problem.grad(f['p'], x)[:, :2].T
    grad_xx = Problem.grad(grad_x, x)[:, 0]
    grad_yy = Problem.grad(grad_y, x)[:, 1]
    return - grad_xx - grad_yy  - f['y'] + 1*(x[:, 0] <= 1) + x[:, 2]*(x[:, 0] > 1)

def control_eq(x, f):
    return alpha*f['u'] - f['p']

def state_eq(x, f):
    grad_x, grad_y = Problem.grad(f['y'], x)[:, :2].T
    grad_xx = Problem.grad(grad_x, x)[:, 0]
    grad_yy = Problem.grad(grad_y, x)[:, 1]
    return - grad_xx - grad_yy  - f['u']

def equation(x, f):
    return state_eq(x, f) + control_eq(x, f) + adjoint_eq(x, f)

laplace = ParametricProblem2D(
        variables=['y', 'u', 'p'], 
        bc=bc,
        domain_bound=np.array([[0, 4],[0, 1]]),
        params_bound=np.array([[0.5, 2.5]]))
laplace.equation = equation
