"""Formulation of the advection problem."""

import torch
from ... import Condition
from ...operator import grad
from ...equation import Equation
from ...domain import CartesianDomain
from ...utils import check_consistency
from ...problem import SpatialProblem, TimeDependentProblem


class AdvectionEquation(Equation):
    """
    Implementation of the advection equation.
    """

    def __init__(self, c):
        """
        Initialization of the :class:`AdvectionEquation`.

        :param c: The advection velocity parameter.
        :type c: float | int
        """
        self.c = c
        check_consistency(self.c, (float, int))

        def equation(input_, output_):
            """
            Implementation of the advection equation.

            :param LabelTensor input_: Input data of the problem.
            :param LabelTensor output_: Output data of the problem.
            :return: The residual of the advection equation.
            :rtype: LabelTensor
            """
            u_x = grad(output_, input_, components=["u"], d=["x"])
            u_t = grad(output_, input_, components=["u"], d=["t"])
            return u_t + self.c * u_x

        super().__init__(equation)


def initial_condition(input_, output_):
    """
    Implementation of the initial condition.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :return: The residual of the initial condition.
    :rtype: LabelTensor
    """
    return output_ - torch.sin(input_.extract("x"))


class AdvectionProblem(SpatialProblem, TimeDependentProblem):
    r"""
    Implementation of the advection problem in the spatial interval
    :math:`[0, 2 \pi]` and temporal interval :math:`[0, 1]`.

    .. seealso::

        **Original reference**: Wang, Sifan, et al. *An expert's guide to
        training physics-informed neural networks*.
        arXiv preprint arXiv:2308.08468 (2023).
        DOI: `arXiv:2308.08468  <https://arxiv.org/abs/2308.08468>`_.
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 2 * torch.pi]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": CartesianDomain({"x": [0, 2 * torch.pi], "t": [0, 1]}),
        "t0": CartesianDomain({"x": [0, 2 * torch.pi], "t": 0.0}),
    }

    conditions = {
        "t0": Condition(domain="t0", equation=Equation(initial_condition)),
    }

    def __init__(self, c=1.0):
        """
        Initialization of the :class:`AdvectionProblem`.

        :param c: The advection velocity parameter.
        :type c: float | int
        """
        super().__init__()

        self.c = c
        check_consistency(self.c, (float, int))

        self.conditions["D"] = Condition(
            domain="D", equation=AdvectionEquation(self.c)
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the advection problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the advection problem.
        :rtype: LabelTensor
        """
        sol = torch.sin(pts.extract("x") - self.c * pts.extract("t"))
        sol.labels = self.output_variables
        return sol
