"""Formulation of the inverse Poisson problem in a square domain."""

import requests
import torch
from io import BytesIO
from ... import Condition
from ... import LabelTensor
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


# URL of the file
url = "https://github.com/mathLab/PINA/raw/refs/heads/master/tutorials/tutorial7/data/pts_0.5_0.5"
# Download the file
response = requests.get(url)
response.raise_for_status()
file_like_object = BytesIO(response.content)
# Set the data
input_data = LabelTensor(
    torch.load(file_like_object, weights_only=False).tensor.detach(),
    ["x", "y", "mu1", "mu2"],
)

# URL of the file
url = "https://github.com/mathLab/PINA/raw/refs/heads/master/tutorials/tutorial7/data/pinn_solution_0.5_0.5"
# Download the file
response = requests.get(url)
response.raise_for_status()
file_like_object = BytesIO(response.content)
# Set the data
output_data = LabelTensor(
    torch.load(file_like_object, weights_only=False).tensor.detach(), ["u"]
)


class InversePoisson2DSquareProblem(SpatialProblem, InverseProblem):
    r"""
    Implementation of the inverse 2-dimensional Poisson problem in the square
    domain :math:`[0, 1] \times [0, 1]`,
    with unknown parameter domain :math:`[-1, 1] \times [-1, 1]`.

    :Example:
        >>> problem = InversePoisson2DSquareProblem()
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
