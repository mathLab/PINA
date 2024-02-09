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


from pina.problem import SpatialProblem
from pina import Condition
from pina.geometry import CartesianDomain
from pina.operators import grad
from pina.equation import Equation, FixedValue
import torch


class FirstOrderODE(SpatialProblem):

    # variable domain range
    x_rng = [0.0, 5.0]
    # field variable
    output_variables = ["y"]
    # create domain
    spatial_domain = CartesianDomain({"x": x_rng})

    # define the ode
    def ode(input_, output_):
        y = output_
        x = input_
        return grad(y, x) + y - x

    # define real solution
    def solution(self, input_):
        x = input_
        return x - 1.0 + 2 * torch.exp(-x)

    # define problem conditions
    conditions = {
        "BC": Condition(
            location=CartesianDomain({"x": x_rng[0]}), equation=FixedValue(1.0)
        ),
        "D": Condition(
            location=CartesianDomain({"x": x_rng}), equation=Equation(ode)
        ),
    }

    truth_solution = solution
