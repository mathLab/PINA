"""Formulation of the Allen Cahn problem."""

import torch
from ... import Condition
from ...operator import grad, laplacian
from ...equation import Equation
from ...domain import CartesianDomain
from ...problem import SpatialProblem, TimeDependentProblem


def allen_cahn_equation(input_, output_):
    """
    Implementation of the Allen Cahn equation.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :return: The residual of the Allen Cahn equation.
    :rtype: LabelTensor
    """
    u_t = grad(output_, input_, components=["u"], d=["t"])
    u_xx = laplacian(output_, input_, components=["u"], d=["x"])
    return u_t - 0.0001 * u_xx + 5 * output_**3 - 5 * output_


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the Allen Cahn problem.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    x = input_.extract("x")
    u_0 = x**2 * torch.cos(torch.pi * x)
    return output_ - u_0


class AllenCahnProblem(TimeDependentProblem, SpatialProblem):
    """
    Implementation of the Allen Cahn problem in the spatial interval
    [-1, 1] and temporal interval [0, 1].
    Reference: https://arxiv.org/pdf/2307.00379.
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": CartesianDomain({"x": [-1, 1], "t": [0, 1]}),
        "t0": CartesianDomain({"x": [-1, 1], "t": 0.0}),
    }

    conditions = {
        "D": Condition(domain="D", equation=Equation(allen_cahn_equation)),
        "t0": Condition(domain="t0", equation=Equation(initial_condition)),
    }
