"""Formulation of the burgers problem."""

import torch
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.domain.cartesian_domain import CartesianDomain
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.condition.condition import Condition
from pina._src.core.utils import check_consistency
from pina._src.equation.equation import Equation
from pina._src.equation.zoo.fixed_value import FixedValue
from pina._src.equation.zoo.burgers_equation import BurgersEquation


def initial_condition(input_, output_):
    """
    Definition of the initial condition of the burgers problem.

    :param LabelTensor input_: The input data of the problem.
    :param LabelTensor output_: The output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    return output_ + torch.sin(torch.pi * input_["x"])


class BurgersProblem(TimeDependentProblem, SpatialProblem):
    r"""
    Implementation of the burgers problem in the spatial interval
    :math:`[-1, 1]` and temporal interval :math:`[0, 1]`.

    .. seealso::

        **Original reference**: Raissi M., Perdikaris P., Karniadakis G. E.
        (2017).
        *Physics Informed Deep Learning (Part I): Data-driven Solutions of
        Nonlinear Partial Differential Equations*.
        DOI: `10.48550 <https://doi.org/10.48550/arXiv.1711.10561>`_.

    :Example:

        >>> problem = BurgersProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "t0": spatial_domain.update(CartesianDomain({"t": 0})),
        "boundary": spatial_domain.partial().update(temporal_domain),
    }

    conditions = {
        "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
        "t0": Condition(domain="t0", equation=Equation(initial_condition)),
    }

    def __init__(self, nu=0):
        """
        Initialization of the :class:`BurgersProblem` class.

        :param nu: The viscosity coefficient.
        :type nu: float | int
        :raises ValueError: If ``nu`` is not a float or an int.
        :raises ValueError: If ``nu`` is negative.
        """
        super().__init__()

        # Check consistency
        check_consistency(nu, (float, int))
        if nu < 0:
            raise ValueError(
                "The viscosity ``nu`` must be a non-negative float or int."
            )

        self.conditions["D"] = Condition(
            domain="D", equation=BurgersEquation(nu)
        )
