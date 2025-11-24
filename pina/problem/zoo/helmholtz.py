"""Formulation of the Helmholtz problem."""

import torch
from ... import Condition
from ...equation import FixedValue, Helmholtz
from ...utils import check_consistency
from ...domain import CartesianDomain
from ...problem import SpatialProblem


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
        "D": CartesianDomain({"x": [-1, 1], "y": [-1, 1]}),
        "g1": CartesianDomain({"x": [-1, 1], "y": 1.0}),
        "g2": CartesianDomain({"x": [-1, 1], "y": -1.0}),
        "g3": CartesianDomain({"x": 1.0, "y": [-1, 1]}),
        "g4": CartesianDomain({"x": -1.0, "y": [-1, 1]}),
    }

    conditions = {
        "g1": Condition(domain="g1", equation=FixedValue(0.0)),
        "g2": Condition(domain="g2", equation=FixedValue(0.0)),
        "g3": Condition(domain="g3", equation=FixedValue(0.0)),
        "g4": Condition(domain="g4", equation=FixedValue(0.0)),
    }

    def __init__(self, alpha=3.0):
        """
        Initialization of the :class:`HelmholtzProblem` class.

        :param alpha: Parameter of the forcing term. Default is 3.0.
        :type alpha: float | int
        """
        super().__init__()
        check_consistency(alpha, (int, float))
        self.alpha = alpha

        def forcing_term(input_):
            """
            Implementation of the forcing term.
            """
            return (
                (1 - 2 * (self.alpha * torch.pi) ** 2)
                * torch.sin(self.alpha * torch.pi * input_.extract("x"))
                * torch.sin(self.alpha * torch.pi * input_.extract("y"))
            )

        self.conditions["D"] = Condition(
            domain="D",
            equation=Helmholtz(self.alpha, forcing_term),
        )

    def solution(self, pts):
        """
        Implementation of the analytical solution of the Helmholtz problem.

        :param LabelTensor pts: Points where the solution is evaluated.
        :return: The analytical solution of the Poisson problem.
        :rtype: LabelTensor
        """
        sol = torch.sin(self.alpha * torch.pi * pts.extract("x")) * torch.sin(
            self.alpha * torch.pi * pts.extract("y")
        )
        sol.labels = self.output_variables
        return sol
