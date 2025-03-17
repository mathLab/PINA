""" Module for Neural Tangent Kernel Class """

from torch.nn import Module
from torch import norm,cat
from .weighting_interface import WeightingInterface
from ..operator import grad
from ..utils import check_consistency

class Neural_Tangent_Kernel(WeightingInterface):
    """
    TODO
    """
    def __init__(self, model, alpha=0.5):
        super().__init__()
        check_consistency(alpha, float)
        
        if alpha<0 or alpha > 1:
            raise ValueError("alpha should be a value between 0 and 1")
        if not isinstance(model, Module):
            raise TypeError("Please pass a valid torch.nn.Module model")
        self.alpha = alpha
        self.model = model
        self.weights = {}
        self.default_value_weights = 1

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
        losses_norm = {}
        for condition in losses:
            losses[condition].backward(retain_graph=True)
            grads=[]
            for param in self.model.parameters():
                grads.append(param.grad.view(-1))
            grads = cat(grads)
            
            print(grads)
            losses_norm[condition]=norm(grads)
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