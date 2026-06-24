"""Module for Neural Tangent Kernel Class"""

import torch
from pina._src.weighting.base_weighting import BaseWeighting
from pina._src.core.utils import check_consistency, in_range


class NeuralTangentKernelWeighting(BaseWeighting):
    """
    The Neural Tangent Kernel (NTK) weighting strategy.

    This weighting scheme dynamically adjusts the contribution of each loss term
    during training by leveraging gradient information with respect to the model
    parameters. For each loss component, the norm of its gradient is computed
    and used to derive relative importance weights. The resulting weights are
    smoothed over time using an exponential moving average controlled by the
    parameter ``alpha``.

    .. seealso::

        **Original reference**: Wang, Sifan, Xinling Yu, and
        Paris Perdikaris. *When and why PINNs fail to train:
        A neural tangent kernel perspective*. Journal of
        Computational Physics 449 (2022): 110768.
        DOI: `10.1016 <https://doi.org/10.1016/j.jcp.2021.110768>`_.

    :Example:

        >>> from pina.weighting import NeuralTangentKernelWeighting
        >>> weighting = NeuralTangentKernelWeighting(alpha=0.5)
        >>> # Typically used within a PINA solver:
        >>> # solver = PINN(problem=problem, weighting=weighting)
    """

    def __init__(self, update_every_n_epochs=1, alpha=0.5):
        """
        Initialization of the :class:`NeuralTangentKernelWeighting` class.

        :param int update_every_n_epochs: The number of training epochs between
            weight updates. If set to 1, the weights are updated at every epoch.
            This parameter is ignored by static weighting schemes.
            Default is ``1``.
        :param float alpha: The parameter controlling the exponential moving
            average of the weights. It must be in the range [0, 1], where a
            value of ``0.0`` means that only the current gradient norms are used
            to compute the weights, and a value of ``1.0`` means that only the
            last saved weights are used. Default is ``0.5``.
        :raises ValueError: If ``alpha`` is not a float.
        :raises ValueError: If ``alpha`` is not between 0.0 and 1.0 (inclusive).
        """
        super().__init__(
            update_every_n_epochs=update_every_n_epochs, aggregator="sum"
        )

        # Check consistency
        check_consistency(alpha, float)
        if not in_range(alpha, [0.0, 1.0], strict=False):
            raise ValueError(
                "The alpha parameter must be between 0.0 and 1.0 (inclusive)."
                f" Got {alpha}."
            )

        # Initialization
        self.alpha = alpha
        self.weights = {}

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

        return {
            condition: self.alpha * self.last_saved_weights().get(condition, 1)
            + (1 - self.alpha) * norms[condition] / sum(norms.values())
            for condition in losses
        }
