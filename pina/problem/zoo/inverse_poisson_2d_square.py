"""Formulation of the inverse Poisson problem in a square domain."""

import warnings
import requests
import torch
from io import BytesIO
from requests.exceptions import RequestException
from ... import Condition
from ... import LabelTensor
from ...operator import laplacian
from ...domain import CartesianDomain
from ...equation import Equation, FixedValue
from ...problem import SpatialProblem, InverseProblem
from ...utils import custom_warning_format

warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=ResourceWarning)


def _load_tensor_from_url(url, labels):
    """
    Downloads a tensor file from a URL and wraps it in a LabelTensor.

    This function fetches a `.pth` file containing tensor data, extracts it,
    and returns it as a LabelTensor using the specified labels. If the file
    cannot be retrieved (e.g., no internet connection), a warning is issued
    and None is returned.

    :param str url: URL to the remote `.pth` tensor file.
    :param list[str] | tuple[str] labels: Labels for the resulting LabelTensor.
    :return: A LabelTensor object if successful, otherwise None.
    :rtype: LabelTensor | None
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        tensor = torch.load(
            BytesIO(response.content), weights_only=False
        ).tensor.detach()
        return LabelTensor(tensor, labels)
    except RequestException as e:
        print(
            "Could not download data for 'InversePoisson2DSquareProblem' "
            f"from '{url}'. "
            f"Reason: {e}. Skipping data loading.",
            ResourceWarning,
        )
        return None


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


# loading data
input_url = (
    "https://github.com/mathLab/PINA/raw/refs/heads/master"
    "/tutorials/tutorial7/data/pts_0.5_0.5"
)
output_url = (
    "https://github.com/mathLab/PINA/raw/refs/heads/master"
    "/tutorials/tutorial7/data/pinn_solution_0.5_0.5"
)
input_data = _load_tensor_from_url(input_url, ["x", "y", "mu1", "mu2"])
output_data = _load_tensor_from_url(output_url, ["u"])


class InversePoisson2DSquareProblem(SpatialProblem, InverseProblem):
    r"""
    Implementation of the inverse 2-dimensional Poisson problem in the square
    domain :math:`[0, 1] \times [0, 1]`,
    with unknown parameter domain :math:`[-1, 1] \times [-1, 1]`.
    The `"data"` condition is added only if the required files are
    downloaded successfully.

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
    }

    if input_data is not None and input_data is not None:
        conditions["data"] = Condition(input=input_data, target=output_data)
