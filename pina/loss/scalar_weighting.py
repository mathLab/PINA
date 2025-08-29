"""Module for the Scalar Weighting."""

from .weighting_interface import WeightingInterface
from ..utils import check_consistency


class _NoWeighting(WeightingInterface):
    """
    Weighting scheme that does not apply any weighting to the losses.
    """

    def aggregate(self, losses):
        """
        Aggregate the losses.

        :param dict losses: The dictionary of losses.
        :return: The aggregated losses.
        :rtype: torch.Tensor
        """
        return sum(losses.values())


class ScalarWeighting(WeightingInterface):
    """
    Weighting scheme that assigns a scalar weight to each loss term.
    """

    def __init__(self, weights):
        """
        Initialization of the :class:`ScalarWeighting` class.

        :param weights: The weights to be assigned to each loss term.
            If a single scalar value is provided, it is assigned to all loss
            terms. If a dictionary is provided, the keys are the conditions and
            the values are the weights. If a condition is not present in the
            dictionary, the default value is used.
        :type weights: float | int | dict
        """
        super().__init__()

        # Check consistency
        check_consistency([weights], (float, dict, int))

        # Weights initialization
        if isinstance(weights, (float, int)):
            self.default_value_weights = weights
            self.weights = {}
        else:
            self.default_value_weights = 1.0
            self.weights = weights

    def aggregate(self, losses):
        """
        Aggregate the losses.

        :param dict losses: The dictionary of losses.
        :return: The aggregated losses.
        :rtype: torch.Tensor
        """
        return sum(
            self.weights.get(condition, self.default_value_weights) * loss
            for condition, loss in losses.items()
        )
