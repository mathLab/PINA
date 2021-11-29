import numpy as np
import torch
from pina.problem import Problem
from pina.segment import Segment
from pina.parametricproblem2d import ParametricProblem2D

bc = (
    (Segment((-1, -1), ( 1, -1)), lambda x: torch.zeros(x.shape[0], 1)),
    (Segment(( 1, -1), ( 1,  1)), lambda x: torch.zeros(x.shape[0], 1)),
    (Segment(( 1,  1), (-1,  1)), lambda x: torch.zeros(x.shape[0], 1)),
    (Segment((-1,  1), (-1, -1)), lambda x: torch.zeros(x.shape[0], 1)),
)

params_domain = np.array([
    [-1.0, 1.0], 
    [-1.0, 1.0]])

def equation(x, fx):
    grad_x, grad_y = Problem.grad(fx, x)[:, :2].T
    grad_xx = Problem.grad(grad_x, x)[:, 0]
    grad_yy = Problem.grad(grad_y, x)[:, 1]
    a = grad_xx + grad_yy - torch.exp(- 2*(x[:, 0] - x[:, 2])**2 - 2*(x[:, 1] - x[:, 3])**2)
    return a


laplace = ParametricProblem2D(bc=bc, domain_bound=params_domain, params_bound=params_domain)

laplace.equation = equation

