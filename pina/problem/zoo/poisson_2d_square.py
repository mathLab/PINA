"""Formulation of the Poisson problem in a square domain."""

import torch
from ...equation import FixedValue, Poisson
from ...problem import SpatialProblem
from ...domain import CartesianDomain
from ... import Condition


def forcing_term(input_):
    """
    Implementation of the forcing term of the Poisson problem.

    :param LabelTensor input_: The points where the forcing term is evaluated.
    :return: The forcing term of the Poisson problem.
    :rtype: LabelTensor
    """
    return (
        torch.sin(input_.extract(["x"]) * torch.pi)
        * torch.sin(input_.extract(["y"]) * torch.pi)
        * (2 * torch.pi**2)
    )


class Poisson2DSquareProblem(SpatialProblem):
    r"""
    Implementation of the 2-dimensional Poisson problem in the square domain
    :math:`[0, 1] \times [0, 1]`.

    :Example:
        >>> problem = Poisson2DSquareProblem()
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
        "D": Condition(domain="D", equation=Poisson(forcing_term=forcing_term)),
    }

    def solution(self, pts):
        """
        Implementation of the analytical solution of the Poisson problem.

        :param LabelTensor pts: The points where the solution is evaluated.
        :return: The analytical solution of the Poisson problem.
        :rtype: LabelTensor
        """
        sol = -(
            torch.sin(pts.extract(["x"]) * torch.pi)
            * torch.sin(pts.extract(["y"]) * torch.pi)
        )
        sol.labels = self.output_variables
        return sol
