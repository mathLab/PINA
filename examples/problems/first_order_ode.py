from pina.problem import SpatialProblem
from pina import Condition, Span
from pina.operators import grad
import torch

# ===================================================== #
#                                                       #
#  This script implements a simple first order ode.     #
#  The FirstOrderODE class is defined inheriting from   #
#  SpatialProblem. We  denote:                          #
#           y --> field variable                        #
#           x --> spatial variable                      #
#                                                       #
# ===================================================== #

class FirstOrderODE(SpatialProblem):

    # variable domain range
    x_rng = [0, 5]
    # field variable
    output_variables = ['y']
    # create domain
    spatial_domain = Span({'x': x_rng})

    # define the ode
    def ode(input_, output_):
        y = output_
        x = input_
        return grad(y, x) + y - x

    # define initial conditions
    def fixed(input_, output_):
        exp_value = 1.
        return output_ - exp_value

    # define real solution
    def solution(self, input_):
        x = input_
        return x - 1.0 + 2*torch.exp(-x)

    # define problem conditions
    conditions = {
        'bc': Condition(location=Span({'x': x_rng[0]}), function=fixed),
        'dd': Condition(location=Span({'x': x_rng}), function=ode),
    }
    truth_solution = solution