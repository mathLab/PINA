"""Module for Self-Adaptive Weighting class."""

import torch
from .weighting_interface import WeightingInterface


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

    def __init__(self, update_every_n_epochs=1):
        """
        Initialization of the :class:`SelfAdaptiveWeighting` class.

        :param int update_every_n_epochs: The number of training epochs between
            weight updates. If set to 1, the weights are updated at every epoch.
            Default is 1.
        """
        super().__init__(update_every_n_epochs=update_every_n_epochs)

    def weights_update(self, losses):
        """
        Update the weighting scheme based on the given losses.

        :param dict losses: The dictionary of losses.
        :return: The updated weights.
        :rtype: dict
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
        return {
            condition: sum(losses_norm.values()) / losses_norm[condition]
            for condition in losses
        }
