"""Formulation of the inverse Poisson problem in a square domain."""

import os
import torch
from ... import Condition
from ...operator import laplacian
from ...domain import CartesianDomain
from ...equation import Equation, FixedValue
from ...problem import SpatialProblem, InverseProblem


def laplace_equation(input_, output_, params_):
    """
    Implementation of the laplace equation.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :param dict params_: Parameters of the problem.
    :return: The residual of the laplace equation.
    :rtype: LabelTensor
    """
    force_term = torch.exp(
        -2 * (input_.extract(["x"]) - params_["mu1"]) ** 2
        - 2 * (input_.extract(["y"]) - params_["mu2"]) ** 2
    )
    delta_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return delta_u - force_term


# Absolute path to the data directory
data_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../../tutorials/tutorial7/data/"
    )
)

# Load input data
input_data = torch.load(
    f=os.path.join(data_dir, "pts_0.5_0.5"), weights_only=False
).extract(["x", "y"])

# Load output data
output_data = torch.load(
    f=os.path.join(data_dir, "pinn_solution_0.5_0.5"), weights_only=False
)


class InversePoisson2DSquareProblem(SpatialProblem, InverseProblem):
    r"""
    Implementation of the inverse 2-dimensional Poisson problem in the square
    domain :math:`[0, 1] \times [0, 1]`,
    with unknown parameter domain :math:`[-1, 1] \times [-1, 1]`.
    """

    output_variables = ["u"]
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    spatial_domain = CartesianDomain({"x": [x_min, x_max], "y": [y_min, y_max]})
    unknown_parameter_domain = CartesianDomain({"mu1": [-1, 1], "mu2": [-1, 1]})

    domains = {
        "g1": CartesianDomain({"x": [x_min, x_max], "y": y_max}),
        "g2": CartesianDomain({"x": [x_min, x_max], "y": y_min}),
        "g3": CartesianDomain({"x": x_max, "y": [y_min, y_max]}),
        "g4": CartesianDomain({"x": x_min, "y": [y_min, y_max]}),
        "D": CartesianDomain({"x": [x_min, x_max], "y": [y_min, y_max]}),
    }

    conditions = {
        "g1": Condition(domain="g1", equation=FixedValue(0.0)),
        "g2": Condition(domain="g2", equation=FixedValue(0.0)),
        "g3": Condition(domain="g3", equation=FixedValue(0.0)),
        "g4": Condition(domain="g4", equation=FixedValue(0.0)),
        "D": Condition(domain="D", equation=Equation(laplace_equation)),
        "data": Condition(input=input_data, target=output_data),
    }
