""" Poisson problem. """

# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Poisson problem. The Poisson class is defined        #
#  inheriting from SpatialProblem. We  denote:          #
#           u --> field variable                        #
#           x,y --> spatial variables                   #
#                                                       #
# ===================================================== #


import torch
from pina.geometry import CartesianDomain
from pina import Condition
from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina.equation import FixedValue, Equation


def laplace_equation(input_, output_):
    """ Laplace equation with sin(pi*x)*sin(pi*y) force term."""
    force_term = torch.sin(input_.extract(["x"]) * torch.pi) * torch.sin(
        input_.extract(["y"]) * torch.pi
    )
    nabla_u = laplacian(output_.extract(["u"]), input_)
    return nabla_u - force_term
class Poisson(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

    conditions = {
        "gamma1": Condition(
            location=CartesianDomain({"x": [0, 1], "y": 1}),
            equation=FixedValue(0.0),
        ),
        "gamma2": Condition(
            location=CartesianDomain({"x": [0, 1], "y": 0}),
            equation=FixedValue(0.0),
        ),
        "gamma3": Condition(
            location=CartesianDomain({"x": 1, "y": [0, 1]}),
            equation=FixedValue(0.0),
        ),
        "gamma4": Condition(
            location=CartesianDomain({"x": 0, "y": [0, 1]}),
            equation=FixedValue(0.0),
        ),
        "D": Condition(
            location=CartesianDomain({"x": [0, 1], "y": [0, 1]}),
            equation=Equation(laplace_equation),
        ),
    }

    def poisson_sol(self, pts):
        return -(
            torch.sin(pts.extract(["x"]) * torch.pi)
            * torch.sin(pts.extract(["y"]) * torch.pi)
        ) / (2 * torch.pi**2)

    truth_solution = poisson_sol
