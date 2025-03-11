"""Formulation of the diffusion-reaction problem."""

import torch
from ... import Condition
from ...domain import CartesianDomain
from ...operator import grad, laplacian
from ...equation import Equation, FixedValue
from ...problem import SpatialProblem, TimeDependentProblem


def diffusion_reaction(input_, output_):
    """
    Implementation of the diffusion-reaction equation.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :return: The residual of the diffusion-reaction equation.
    :rtype: LabelTensor
    """
    x = input_.extract("x")
    t = input_.extract("t")
    u_t = grad(output_, input_, components=["u"], d=["t"])
    u_xx = laplacian(output_, input_, components=["u"], d=["x"])
    r = torch.exp(-t) * (
        1.5 * torch.sin(2 * x)
        + (8 / 3) * torch.sin(3 * x)
        + (15 / 4) * torch.sin(4 * x)
        + (63 / 8) * torch.sin(8 * x)
    )
    return u_t - u_xx - r


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the diffusion-reaction problem.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    x = input_.extract("x")
    u_0 = (
        torch.sin(x)
        + (1 / 2) * torch.sin(2 * x)
        + (1 / 3) * torch.sin(3 * x)
        + (1 / 4) * torch.sin(4 * x)
        + (1 / 8) * torch.sin(8 * x)
    )
    return output_ - u_0


class DiffusionReactionProblem(TimeDependentProblem, SpatialProblem):
    """
    Implementation of the diffusion-reaction problem in the spatial interval
    [-pi, pi] and temporal interval [0, 1].
    Reference: https://www.arxiv.org/pdf/2502.04917.
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-torch.pi, torch.pi]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": CartesianDomain({"x": [-torch.pi, torch.pi], "t": [0, 1]}),
        "g1": CartesianDomain({"x": -torch.pi, "t": [0, 1]}),
        "g2": CartesianDomain({"x": torch.pi, "t": [0, 1]}),
        "t0": CartesianDomain({"x": [-torch.pi, torch.pi], "t": 0.0}),
    }

    conditions = {
        "D": Condition(domain="D", equation=Equation(diffusion_reaction)),
        "g1": Condition(domain="g1", equation=FixedValue(0.0)),
        "g2": Condition(domain="g2", equation=FixedValue(0.0)),
        "t0": Condition(domain="t0", equation=Equation(initial_condition)),
    }

    def solution(self, pts):
        """
        Implementation of the analytical solution of the diffusion-reaction
        problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the diffusion-reaction problem.
        :rtype: LabelTensor
        """
        t = pts.extract("t")
        x = pts.extract("x")
        sol = torch.exp(-t) * (
            torch.sin(x)
            + (1 / 2) * torch.sin(2 * x)
            + (1 / 3) * torch.sin(3 * x)
            + (1 / 4) * torch.sin(4 * x)
            + (1 / 8) * torch.sin(8 * x)
        )
        sol.labels = self.output_variables
        return sol
