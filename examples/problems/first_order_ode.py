""" Simple ODE problem. """

# ===================================================== #
#                                                       #
#  This script implements a simple first order ode.     #
#  The FirstOrderODE class is defined inheriting from   #
#  SpatialProblem. We  denote:                          #
#           y --> field variable                        #
#           x --> spatial variable                      #
#                                                       #
#  The equation is:                                     #
#           dy(x)/dx + y(x) = x                         #
#                                                       #
# ===================================================== #

import torch

from pina.problem import SpatialProblem
from pina import Condition
from pina.geometry import CartesianDomain
from pina.operators import grad
from pina.equation import Equation, FixedValue


def ode(input_, output_):
    """ First order ODE: dy/dx + y = x."""
    y = output_
    x = input_
    return grad(y, x) + y - x

class FirstOrderODE(SpatialProblem):

    x_rng = [0.0, 5.0]
    output_variables = ["y"]
    spatial_domain = CartesianDomain({"x": x_rng})

    # define problem conditions
    conditions = {
        "BC": Condition(
            location=CartesianDomain({"x": x_rng[0]}), equation=FixedValue(1.0)
        ),
        "D": Condition(
            location=CartesianDomain({"x": x_rng}), equation=Equation(ode)
        ),
    }

    def solution(self, input_):
        """ Truth solution """
        x = input_
        return x - 1.0 + 2 * torch.exp(-x)

    truth_solution = solution
