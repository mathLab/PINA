""" Navier Stokes Problem """

import torch
from pina.problem import SpatialProblem
from pina.operators import laplacian, grad, div
from pina import Condition, LabelTensor
from pina.geometry import CartesianDomain
from pina.equation import Equation

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

    # problem condition statement
    conditions = {
        'gamma_top': Condition(location=CartesianDomain({'x': [-2, 2], 'y':  1}), equation=Equation(wall)),
        'gamma_bot': Condition(location=CartesianDomain({'x': [-2, 2], 'y': -1}), equation=Equation(wall)),
        'gamma_out': Condition(location=CartesianDomain({'x':  2, 'y': [-1, 1]}), equation=Equation(outlet)),
        'gamma_in':  Condition(location=CartesianDomain({'x': -2, 'y': [-1, 1]}), equation=Equation(inlet)),
        'D': Condition(location=CartesianDomain({'x': [-2, 2], 'y': [-1, 1]}), equation=[Equation(momentum), Equation(continuity)])
    }
