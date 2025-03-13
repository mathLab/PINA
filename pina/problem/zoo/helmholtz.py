"""Formulation of the Helmholtz problem."""

import torch
from ... import Condition
from ...operator import laplacian
from ...domain import CartesianDomain
from ...problem import SpatialProblem
from ...utils import check_consistency
from ...equation import Equation, FixedValue


class HelmholtzEquation(Equation):
    """
    Implementation of the Helmholtz equation.
    """

    def __init__(self, alpha):
        """
        Initialization of the :class:`HelmholtzEquation` class.

        :param alpha: Parameter of the forcing term.
        :type alpha: float | int
        """
        self.alpha = alpha
        check_consistency(alpha, (int, float))

        def equation(input_, output_):
            """
            Implementation of the Helmholtz equation.

            :param LabelTensor input_: Input data of the problem.
            :param LabelTensor output_: Output data of the problem.
            :return: The residual of the Helmholtz equation.
            :rtype: LabelTensor
            """
            lap = laplacian(output_, input_, components=["u"], d=["x", "y"])
            q = (
                (1 - 2 * (self.alpha * torch.pi) ** 2)
                * torch.sin(self.alpha * torch.pi * input_.extract("x"))
                * torch.sin(self.alpha * torch.pi * input_.extract("y"))
            )
            return lap + output_ - q

        super().__init__(equation)


class HelmholtzProblem(SpatialProblem):
    r"""
    Implementation of the Helmholtz problem in the square domain
    :math:`[-1, 1] \times [-1, 1]`.

    .. seealso::
        **Original reference**: Si, Chenhao, et al. *Complex Physics-Informed
        Neural Network.* arXiv preprint arXiv:2502.04917 (2025).
        DOI: `arXiv:2502.04917 <https://arxiv.org/abs/2502.04917>`_.
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

        :param alpha: Parameter of the forcing term.
        :type alpha: float | int
        """
        super().__init__()

        self.alpha = alpha
        check_consistency(alpha, (int, float))

        self.conditions["D"] = Condition(
            domain="D", equation=HelmholtzEquation(self.alpha)
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
