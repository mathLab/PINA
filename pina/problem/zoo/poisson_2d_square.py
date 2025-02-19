"""Definition of the Poisson problem on a square domain."""

import torch
from ..spatial_problem import SpatialProblem
from ...operator import laplacian
from ... import Condition
from ...domain import CartesianDomain
from ...equation.equation import Equation
from ...equation.equation_factory import FixedValue


def laplace_equation(input_, output_):
    """
    Implementation of the laplace equation.
    """
    force_term = torch.sin(input_.extract(["x"]) * torch.pi) * torch.sin(
        input_.extract(["y"]) * torch.pi
    )
    delta_u = laplacian(output_.extract(["u"]), input_)
    return delta_u - force_term


my_laplace = Equation(laplace_equation)


class Poisson2DSquareProblem(SpatialProblem):
    """
    Implementation of the 2-dimensional Poisson problem on a square domain.
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

    domains = {
        "D": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
        "g1": CartesianDomain({"x": [0, 1], "y": 1}),
        "g2": CartesianDomain({"x": [0, 1], "y": 0}),
        "g3": CartesianDomain({"x": 1, "y": [0, 1]}),
        "g4": CartesianDomain({"x": 0, "y": [0, 1]}),
    }

    conditions = {

        'g1': Condition(domain='g1', equation=FixedValue(0.0)),
        'g2': Condition(domain='g2', equation=FixedValue(0.0)),
        'g3': Condition(domain='g3', equation=FixedValue(0.0)),
        'g4': Condition(domain='g4', equation=FixedValue(0.0)),
        'D': Condition(domain='D', equation=my_laplace),
    }

    def solution(self, pts):
        return -(torch.sin(pts.extract(['x']) * torch.pi) *
                 torch.sin(pts.extract(['y']) * torch.pi))

