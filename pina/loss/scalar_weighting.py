""" Module for Loss Interface """

from .weighting_interface import WeightingInterface
from ..utils import check_consistency


class _NoWeighting(WeightingInterface):
    def aggregate(self, losses):
        return sum(losses.values())

class ScalarWeighting(WeightingInterface):
    """
    TODO
    """
    def __init__(self, weights):
        super().__init__()
        check_consistency([weights], (float, dict, int))
        if isinstance(weights, (float, int)):
            self.default_value_weights = weights
            self.weights = {}
        else:
            self.default_value_weights = 1
            self.weights = weights

    def aggregate(self, losses):
        """
        Aggregate the losses.

        :param dict(torch.Tensor) input: The dictionary of losses.
        :return: The losses aggregation. It should be a scalar Tensor.
        :rtype: torch.Tensor
        """
        # all conditions must have a single value
        for condition in self.condition_names:
            self.weights.setdefault(condition, self.default_value_weights)
        # update
        weighted_losses = {
            condition: self.weights[condition] * losses[condition] 
            for condition in losses
        }
        return sum(weighted_losses.values())