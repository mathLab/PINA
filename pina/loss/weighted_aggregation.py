""" Module for Loss Interface """

from .weightning_interface import weightningInterface
from torch import norm
from pina.operators import grad

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


    def NTK_weighting(self, losses, alpha = 0.5):
        """
        Weights the losses according to the Neural Tangent Kernel 

        :param dict(torch.Tensor) input: The dictionary of losses.
        :param alpha(float) input: The parameter alpha that regulates the moving average
        between old and new weights. 
        :return: The losses aggregation. It should be a scalar Tensor.
        :rtype: torch.Tensor

        Reference:
        Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023). 
        An expert's guide to training physics-informed neural networks. 
        arXiv preprint arXiv:2308.08468.
        """
        self.aggr = 'sum'

        if self.weights:
            
            losses_norm = {
                condition: norm(grad(losses[condition])) for condition in losses
            }
            self.weights = {
                condition: alpha* self.weights[condition] + 
                (1- alpha)*losses_norm[condition]/sum(losses_norm.values())
                for condition in losses
            }
            
        else: 
            self.weights = {
                condition: 1 for condition in losses}
        return self.aggregate(losses)