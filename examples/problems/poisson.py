""" Poisson equation example. """
import numpy as np
import torch

from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina import Condition, Span

# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Poisson problem. The Poisson class is defined        #
#  inheriting from SpatialProblem. We  denote:          #
#           u --> field variable                        #
#           x,y --> spatial variables                   #
#                                                       #
# ===================================================== #


class Poisson(SpatialProblem):

    # assign output/ spatial variables
    output_variables = ['u']
    spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})

    # define the laplace equation
    def laplace_equation(input_, output_):
        force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                      torch.sin(input_.extract(['y'])*torch.pi))
        delta_u = laplacian(output_.extract(['u']), input_)
        return delta_u - force_term

    # define nill dirichlet boundary conditions
    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    # problem condition statement
    conditions = {
        'gamma1': Condition(location=Span({'x': [0, 1], 'y':  1}), function=nil_dirichlet),
        'gamma2': Condition(location=Span({'x': [0, 1], 'y': 0}), function=nil_dirichlet),
        'gamma3': Condition(location=Span({'x':  1, 'y': [0, 1]}),function=nil_dirichlet),
        'gamma4': Condition(location=Span({'x': 0, 'y': [0, 1]}), function=nil_dirichlet),
        'D': Condition(location=Span({'x': [0, 1], 'y': [0, 1]}), function=laplace_equation),
    }

    # real poisson solution
    def poisson_sol(self, pts):
        return -(
            torch.sin(pts.extract(['x'])*torch.pi) *
            torch.sin(pts.extract(['y'])*torch.pi)
        )/(2*torch.pi**2)
        # return -(np.sin(x*np.pi)*np.sin(y*np.pi))/(2*np.pi**2)

    truth_solution = poisson_sol
