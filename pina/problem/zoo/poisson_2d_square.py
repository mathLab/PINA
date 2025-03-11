"""Formulation of the Poisson problem in a square domain."""

import torch
from ... import Condition
from ...operator import laplacian
from ...problem import SpatialProblem
from ...domain import CartesianDomain
from ...equation import Equation, FixedValue


def laplace_equation(input_, output_):
    """
    Implementation of the laplace equation.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :return: The residual of the laplace equation.
    :rtype: LabelTensor
    """
    force_term = (
        torch.sin(input_.extract(["x"]) * torch.pi)
        * torch.sin(input_.extract(["y"]) * torch.pi)
        * (2 * torch.pi**2)
    )
    delta_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return delta_u - force_term


class Poisson2DSquareProblem(SpatialProblem):
    """
    Implementation of the 2-dimensional Poisson problem in a square domain.
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

    domains = {
        "D": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
        "g1": CartesianDomain({"x": [0, 1], "y": 1.0}),
        "g2": CartesianDomain({"x": [0, 1], "y": 0.0}),
        "g3": CartesianDomain({"x": 1.0, "y": [0, 1]}),
        "g4": CartesianDomain({"x": 0.0, "y": [0, 1]}),
    }

    conditions = {
        "g1": Condition(domain="g1", equation=FixedValue(0.0)),
        "g2": Condition(domain="g2", equation=FixedValue(0.0)),
        "g3": Condition(domain="g3", equation=FixedValue(0.0)),
        "g4": Condition(domain="g4", equation=FixedValue(0.0)),
        "D": Condition(domain="D", equation=Equation(laplace_equation)),
    }

    def solution(self, pts):
        """
        Implementation of the analytical solution of the Poisson problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the Poisson problem.
        :rtype: LabelTensor
        """
        sol = -(
            torch.sin(pts.extract(["x"]) * torch.pi)
            * torch.sin(pts.extract(["y"]) * torch.pi)
        )
        sol.labels = self.output_variables
        return sol
