"""Module for Neural Tangent Kernel Class"""

import torch
from .weighting_interface import WeightingInterface
from ..utils import check_consistency


class NeuralTangentKernelWeighting(WeightingInterface):
    """
    A neural tangent kernel scheme for weighting different losses to
    boost the convergence.

    .. seealso::

        **Original reference**: Wang, Sifan, Xinling Yu, and
        Paris Perdikaris. *When and why PINNs fail to train:
        A neural tangent kernel perspective*. Journal of
        Computational Physics 449 (2022): 110768.
        DOI: `10.1016 <https://doi.org/10.1016/j.jcp.2021.110768>`_.

    """

    def __init__(self, alpha=0.5):
        """
        Initialization of the :class:`NeuralTangentKernelWeighting` class.

        :param float alpha: The alpha parameter.
        :raises ValueError: If ``alpha`` is not between 0 and 1 (inclusive).
        """
        super().__init__()

        # Check consistency
        check_consistency(alpha, float)
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha should be a value between 0 and 1")

        # Initialize parameters
        self.alpha = alpha
        self.weights = {}
        self.default_value_weights = 1.0

    def aggregate(self, losses):
        """
        Weight the losses according to the Neural Tangent Kernel algorithm.

        :param dict(torch.Tensor) input: The dictionary of losses.
        :return: The aggregation of the losses. It should be a scalar Tensor.
        :rtype: torch.Tensor
        """
        # Define a dictionary to store the norms of the gradients
        losses_norm = {}

        # Compute the gradient norms for each loss component
        for condition, loss in losses.items():
            loss.backward(retain_graph=True)
            grads = torch.cat(
                [p.grad.flatten() for p in self.solver.model.parameters()]
            )
            losses_norm[condition] = grads.norm()

        # Update the weights
        self.weights = {
            condition: self.alpha
            * self.weights.get(condition, self.default_value_weights)
            + (1 - self.alpha)
            * losses_norm[condition]
            / sum(losses_norm.values())
            for condition in losses
        }

        return sum(
            self.weights[condition] * loss for condition, loss in losses.items()
        )
