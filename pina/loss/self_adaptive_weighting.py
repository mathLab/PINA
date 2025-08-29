"""Module for Self-Adaptive Weighting class."""

import torch
from .weighting_interface import WeightingInterface
from ..utils import check_positive_integer


class SelfAdaptiveWeighting(WeightingInterface):
    """
    A self-adaptive weighting scheme to tackle the imbalance among the loss
    components. This formulation equalizes the gradient norms of the losses,
    preventing bias toward any particular term during training.

    .. seealso::

        **Original reference**:
        Wang, S., Sankaran, S., Stinis., P., Perdikaris, P. (2025).
        *Simulating Three-dimensional Turbulence with Physics-informed Neural
        Networks*.
        DOI: `arXiv preprint arXiv:2507.08972.
        <https://arxiv.org/abs/2507.08972>`_

    """

    def __init__(self, k=100):
        """
        Initialization of the :class:`SelfAdaptiveWeighting` class.

        :param int k: The number of epochs after which the weights are updated.
            Default is 100.

        :raises ValueError: If ``k`` is not a positive integer.
        """
        super().__init__()

        # Check consistency
        check_positive_integer(value=k, strict=True)

        # Initialize parameters
        self.k = k
        self.weights = {}
        self.default_value_weights = 1.0

    def aggregate(self, losses):
        """
        Weight the losses according to the self-adaptive algorithm.

        :param dict(torch.Tensor) losses: The dictionary of losses.
        :return: The aggregation of the losses. It should be a scalar Tensor.
        :rtype: torch.Tensor
        """
        # If weights have not been initialized, set them to 1
        if not self.weights:
            self.weights = {
                condition: self.default_value_weights for condition in losses
            }

        # Update every k epochs
        if self.solver.trainer.current_epoch % self.k == 0:

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
                condition: sum(losses_norm.values()) / losses_norm[condition]
                for condition in losses
            }

        return sum(
            self.weights[condition] * loss for condition, loss in losses.items()
        )
