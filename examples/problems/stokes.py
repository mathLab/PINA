""" Steady Stokes Problem """

import torch
from pina.problem import SpatialProblem
from pina.operators import laplacian, grad, div
from pina import Condition, LabelTensor
from pina.domain import CartesianDomain
from pina.equation import SystemEquation, Equation

# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Stokes problem. The Stokes class is defined          #
#  inheriting from SpatialProblem. We  denote:          #
#           ux --> field variable velocity along x      #
#           uy --> field variable velocity along y      #
#           p --> field variable pressure               #
#           x,y --> spatial variables                   #
#                                                       #
# ===================================================== #

class Stokes(SpatialProblem):

    # assign output/ spatial variables
    output_variables = ['ux', 'uy', 'p']
    spatial_domain = CartesianDomain({'x': [-2, 2], 'y': [-1, 1]})

    # define the momentum equation
    def momentum(input_, output_):
        delta_ = torch.hstack((LabelTensor(laplacian(output_.extract(['ux']), input_), ['x']),
            LabelTensor(laplacian(output_.extract(['uy']), input_), ['y'])))
        return - delta_ + grad(output_.extract(['p']), input_)

    def continuity(input_, output_):
        return div(output_.extract(['ux', 'uy']), input_)

    # define the inlet velocity
    def inlet(input_, output_):
        value = 2 * (1 - input_.extract(['y'])**2)
        return output_.extract(['ux']) - value
    
    # define the outlet pressure
    def outlet(input_, output_):
        value = 0.0
        return output_.extract(['p']) - value

    # define the wall condition
    def wall(input_, output_):
        value = 0.0
        return output_.extract(['ux', 'uy']) - value

    domains = {
        'gamma_top': CartesianDomain({'x': [-2, 2], 'y':  1}),
        'gamma_bot': CartesianDomain({'x': [-2, 2], 'y': -1}),
        'gamma_out': CartesianDomain({'x':  2, 'y': [-1, 1]}),
        'gamma_in':  CartesianDomain({'x': -2, 'y': [-1, 1]}),
        'D': CartesianDomain({'x': [-2, 2], 'y': [-1, 1]})
    }

    # problem condition statement
    conditions = {
        'gamma_top': Condition(domain='gamma_top', equation=Equation(wall)),
        'gamma_bot': Condition(domain='gamma_bot', equation=Equation(wall)),
        'gamma_out': Condition(domain='gamma_out', equation=Equation(outlet)),
        'gamma_in':  Condition(domain='gamma_in', equation=Equation(inlet)),
        'D': Condition(domain='D', equation=SystemEquation([momentum, continuity]))
    }
