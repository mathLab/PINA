import numpy as np
import scipy.io
import torch

from pina.problem import Problem
from pina.segment import Segment
from pina.cube import Cube
from pina.tdproblem1d import TimeDepProblem1D

def tmp_grad(output_, input_):
    return torch.autograd.grad(
            output_, 
            input_.tensor, 
            grad_outputs=torch.ones(output_.size()).to(
                dtype=input_.tensor.dtype, 
                device=input_.tensor.device), 
            create_graph=True, retain_graph=True, allow_unused=True)[0]

class Burgers1D(TimeDepProblem1D):

    def __init__(self):


        def burger_equation(input_, output_):

            grad_u = self.grad(output_['u'], input_)
            grad_x, grad_t = tmp_grad(output_['u'], input_).T
            gradgrad_u_x = self.grad(grad_u['x'], input_)
            grad_xx = tmp_grad(grad_x, input_)[:, 0]
            #print(grad_t, grad_u['t'])

            #rrrr
            return grad_u['t'] + output_['u']*grad_u['x'] - (0.01/torch.pi)*gradgrad_u_x['x']


        def nil_dirichlet(input_, output_):
            u_expected = 0.0
            return output_['u'] - u_expected

        def initial_condition(input_, output_):
            u_expected = -torch.sin(torch.pi*input_['x'])
            return output_['u'] - u_expected



        self.conditions = {
            'gamma1': {'location': Segment((-1, 0), (-1, 1)), 'func': nil_dirichlet},
            'gamma2': {'location': Segment(( 1, 0), ( 1, 1)), 'func': nil_dirichlet},
            'initia': {'location': Segment((-1, 0), ( 1, 0)), 'func': initial_condition},
            'D': {'location': Cube([[-1, 1],[0,1]]), 'func': burger_equation}
        }

        self.input_variables = ['x', 't']
        self.output_variables = ['u']
        self.spatial_domain = Cube([[0, 1]])
        self.temporal_domain = Cube([[0, 1]])

bc = (
    (-1, lambda x: torch.zeros(x.shape[0], 1)),
    ( 1, lambda x: torch.zeros(x.shape[0], 1))
)

initial = lambda x: -np.sin(np.pi*x[:,0]).reshape(-1, 1)

def equation(x, fx):
    grad_x, grad_t = Problem.grad(fx, x).T
    grad_xx = Problem.grad(grad_x, x)[:, 0]
    a = grad_t + fx.flatten()*grad_x - (0.01/torch.pi)*grad_xx
    return a


burgers = TimeDepProblem1D(bc=bc, initial=initial, tend=1, domain_bound=[-1, 1])
burgers.equation = equation

# read data for errors and plots
data = scipy.io.loadmat('Data/burgers_shock.mat')
data_solution = {'grid': np.meshgrid(data['x'], data['t']), 'grid_solution': data['usol'].T}
burgers.data_solution = data_solution
