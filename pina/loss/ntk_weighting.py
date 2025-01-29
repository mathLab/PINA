""" Module for Loss Interface """

from torch import norm
from .weighting_interface import WeightingInterface
from ..operators import grad
from ..utils import check_consistency


class WeightedAggregation(WeightingInterface):
    """
    TODO
    """
    def __init__(self, alpha):
        super().__init__()
        check_consistency(alpha, float)
        self.alpha = alpha

    def aggregate(self, losses):
        """
        Weights the losses according to the Neural Tangent Kernel 

        :param dict(torch.Tensor) input: The dictionary of losses.
        :param alpha(float) input: The parameter alpha that regulates the moving average
        between old and new weights. 
        :return: The losses aggregation. It should be a scalar Tensor.
        :rtype: torch.Tensor

        Reference: TODO
        Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023). 
        An expert's guide to training physics-informed neural networks. 
        arXiv preprint arXiv:2308.08468.
        """            
        losses_norm = {
            condition: norm(grad(losses[condition])) for condition in losses
        }
        self.weights = {
            condition: 
            self.alpha * self.weights.get(
                condition, self.default_value_weights) + 
            (1- self.alpha)*losses_norm[condition]/sum(losses_norm.values())
            for condition in losses
        }
        return sum(
            self.weights.get(condition, self.default_value_weights) * loss for
            condition, loss in losses.items()
        )