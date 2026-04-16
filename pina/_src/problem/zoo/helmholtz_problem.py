"""Formulation of the Helmholtz problem."""

import torch
from pina._src.condition.condition import Condition
from pina._src.equation.equation_factory import FixedValue
from pina._src.equation.zoo.helmholtz_equation import HelmholtzEquation
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.core.utils import check_consistency
from pina._src.domain.cartesian_domain import CartesianDomain


class HelmholtzProblem(SpatialProblem):
    r"""
    Implementation of the Helmholtz problem in the square domain
    :math:`[-1, 1] \times [-1, 1]`.

    .. seealso::

        **Original reference**: Si, Chenhao, et al. *Complex Physics-Informed
        Neural Network.* arXiv preprint arXiv:2502.04917 (2025).
        DOI: `arXiv:2502.04917 <https://arxiv.org/abs/2502.04917>`_.

    :Example:

        >>> problem = HelmholtzProblem()
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1], "y": [-1, 1]})

    domains = {
        "D": spatial_domain,
        "boundary": spatial_domain.partial(),
    }

    conditions = {
        "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
    }

    def __init__(self, k=1.0, alpha_x=1, alpha_y=4):
        """
        Initialization of the :class:`HelmholtzProblem` class.

        :param k: The squared wavenumber. Default is ``1.0``.
        :type k: float | int
        :param int alpha_x: The frequency in the x-direction. Default is ``1``.
        :param int alpha_y: The frequency in the y-direction. Default is ``4``.
        """
        super().__init__()
        check_consistency(k, (int, float))
        check_consistency(alpha_x, int)
        check_consistency(alpha_y, int)
        self.k = k
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        def forcing_term(input_):
            """
            Implementation of the forcing term.
            """
            x, y, pi = input_["x"], input_["y"], torch.pi
            factor = (self.alpha_x**2 + self.alpha_y**2) * pi**2
            return (
                (self.k - factor)
                * torch.sin(self.alpha_x * pi * x)
                * torch.sin(self.alpha_y * pi * y)
            )

        self.conditions["D"] = Condition(
            domain="D",
            equation=HelmholtzEquation(self.k, forcing_term),
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the Helmholtz problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the Helmholtz problem.
        :rtype: LabelTensor
        """
        x, y, pi = pts["x"], pts["y"], torch.pi
        sol = torch.sin(self.alpha_x * pi * x) * torch.sin(
            self.alpha_y * pi * y
        )
        sol.labels = self.output_variables
        return sol
