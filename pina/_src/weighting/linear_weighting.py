"""Module for the Linear Weighting class."""

from pina._src.weighting.base_weighting import BaseWeighting
from pina._src.core.utils import check_consistency, check_positive_integer


class LinearWeighting(BaseWeighting):
    """
    Weighting strategy based on linear interpolation over training epochs.

    This scheme progressively adjusts the weights assigned to each loss term,
    transitioning from a set of initial values to corresponding final values.
    The update follows a linear schedule and is applied at each epoch until the
    specified target epoch is reached.

    :Example:

        >>> import torch
        >>> from pina.weighting import LinearWeighting
        >>> initial = {"residual": 1.0, "data": 0.5}
        >>> final = {"residual": 0.1, "data": 1.0}
        >>> weighting = LinearWeighting(
        ...     initial_weights=initial,
        ...     final_weights=final,
        ...     target_epoch=100,
        ... )
        >>> # Weights are interpolated linearly over the first 100 epochs
        >>> losses = {"residual": torch.tensor(0.1), "data": torch.tensor(0.2)}
        >>> # The update_weights method is called internally via aggregate
    """

    def __init__(self, initial_weights, final_weights, target_epoch):
        """
        Initialization of the :class:`LinearWeighting` class.

        :param dict initial_weights: The mapping of loss identifiers to their
            initial weights at the start of training. Keys represent loss terms
            (e.g., conditions) and values are the corresponding weights. Loss
            terms not explicitly specified default to a weight of ``1``.
        :param dict final_weights: The mapping of loss identifiers to their
            target weights at the specified ``target_epoch``. Keys must match
            those of ``initial_weights``. Loss terms not explicitly specified
            default to a weight of ``1``.
        :param int target_epoch: The epoch at which the weights reach their
            final values. The interpolation progresses linearly from epoch ``0``
            to ``target_epoch``. After ``target_epoch``, the weights remain
            constant at their final values.
        :raises ValueError: If ``initial_weights`` is not a dictionary.
        :raises ValueError: If ``final_weights`` is not a dictionary.
        :raises ValueError: If ``target_epoch`` is not a positive integer.
        :raises ValueError: If the keys of the two dictionaries are not
            consistent.
        """
        super().__init__(update_every_n_epochs=1, aggregator="sum")

        # Check consistency
        check_consistency([initial_weights, final_weights], dict)
        check_positive_integer(value=target_epoch, strict=True)

        # Check that the keys of the two dictionaries match
        if initial_weights.keys() != final_weights.keys():
            raise ValueError(
                "The keys of the provided dictionaries for initial and final "
                f"weights must match. Got {initial_weights.keys()} as initial "
                f"weight keys and {final_weights.keys()} as final weight keys."
            )

        # Initialization
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.target_epoch = target_epoch

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
        # Determine the progress towards the target epoch
        epoch = min(self.solver.trainer.current_epoch, self.target_epoch)
        progress = epoch / self.target_epoch

        return {
            condition: self.initial_weights[condition]
            + (self.final_weights[condition] - self.initial_weights[condition])
            * progress
            for condition in losses.keys()
        }
