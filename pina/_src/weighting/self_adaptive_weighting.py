"""Module for Self-Adaptive Weighting class."""

import torch
from pina._src.weighting.base_weighting import BaseWeighting


class SelfAdaptiveWeighting(BaseWeighting):
    """
    The self-adaptive weighting strategy based on gradient norm balancing.

    This scheme dynamically adjusts the weights assigned to each loss term by
    computing the norm of their gradients with respect to the model parameters.
    The resulting weights are chosen to counterbalance disparities in gradient
    magnitudes, promoting a more uniform contribution of all loss components
    during optimization.

    In practice, loss terms with smaller gradient norms are assigned larger
    weights, while those with larger gradients are down-weighted. This helps
    mitigate training imbalance and prevents dominance of specific loss terms.

    .. seealso::

        **Original reference**:
        Wang, S., Sankaran, S., Stinis., P., Perdikaris, P. (2025).
        *Simulating Three-dimensional Turbulence with Physics-informed Neural
        Networks*.
        DOI: `arXiv preprint arXiv:2507.08972.
        <https://arxiv.org/abs/2507.08972>`_

    :Example:

        >>> from pina.weighting import SelfAdaptiveWeighting
        >>> weighting = SelfAdaptiveWeighting(update_every_n_epochs=5)
        >>> # Typically used within a PINA solver:
        >>> # solver = PINN(problem=problem, weighting=weighting)
    """

    def __init__(self, update_every_n_epochs=1):
        """
        Initialization of the :class:`SelfAdaptiveWeighting` class.

        :param int update_every_n_epochs: The number of training epochs between
            weight updates. If set to 1, the weights are updated at every epoch.
            This parameter is ignored by static weighting schemes.
            Default is ``1``.
        """
        super().__init__(
            update_every_n_epochs=update_every_n_epochs, aggregator="sum"
        )

    def update_weights(self, losses):
        """
        Update the weights based on the current losses.

        This method defines how the weighting strategy adapts over time. It is
        responsible for computing and storing updated weights that will be used
        during aggregation.

        :param dict losses: The mapping from loss names to loss tensors.
        :return: The updated weights.
        :rtype: dict
        """
        # Get model parameters and define a dictionary to store the norms
        params = [p for p in self.solver.model.parameters() if p.requires_grad]
        norms = {}

        # Iterate over conditions
        for condition, loss in losses.items():

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )

            # Compute norms
            norms[condition] = torch.cat(
                [g.flatten() for g in grads if g is not None]
            ).norm()

        # Update the weights
        return {
            condition: sum(norms.values()) / norms[condition]
            for condition in losses
        }
