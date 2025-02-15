""" Module for Loss Interface """

from .weightning_interface import weightningInterface


class WeightedAggregation(WeightningInterface):
    """
    TODO
    """
    def __init__(self, aggr='mean', weights=None):
        self.aggr = aggr
        self.weights = weights

    def aggregate(self, losses):
        """
        Aggregate the losses.

        :param dict(torch.Tensor) input: The dictionary of losses.
        :return: The losses aggregation. It should be a scalar Tensor.
        :rtype: torch.Tensor
        """
        if self.weights:
            weighted_losses = {
                condition: self.weights[condition] * losses[condition] 
                for condition in losses
            }
        else:
            weighted_losses = losses

        if self.aggr == 'mean':
            return sum(weighted_losses.values()) / len(weighted_losses)
        elif self.aggr == 'sum':
            return sum(weighted_losses.values())
        else:
            raise ValueError(self.aggr + " is not valid for aggregation.")
