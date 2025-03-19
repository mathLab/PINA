"""Module for Neural Tangent Kernel Class"""

import torch
from torch.nn import Module
from .weighting_interface import WeightingInterface
from ..utils import check_consistency


class NeuralTangentKernelWeighting(WeightingInterface):
    """
    A neural tangent kernel scheme for weighting different losses to
    boost the convergence.

    .. seealso::

        **Original reference**: Wang, Sifan, Xinling Yu, and
        Paris Perdikaris. *When and why PINNs fail to train:
        A neural tangent kernel perspective*. Journal of
        Computational Physics 449 (2022): 110768.
        DOI: `10.1016/j.jcp.2021.110768 <https://doi.org/10.1016/j.jcp.2021.110768>`_.



    """

    def __init__(self, model, alpha=0.5):
        """
        Initialization of the :class:`NeuralTangentKernelWeighting` class.

        :param torch.nn.Module model: The neural network model.
        :param float alpha: The alpha parameter.
        """

        super().__init__()
        check_consistency(alpha, float)
        check_consistency(model, Module)
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha should be a value between 0 and 1")
        self.alpha = alpha
        self.model = model
        self.weights = {}
        self.default_value_weights = 1

    def aggregate(self, losses):
        """
        Weights the losses according to the Neural Tangent Kernel
        algorithm.

        :param dict(torch.Tensor) input: The dictionary of losses.
        :return: The losses aggregation. It should be a scalar Tensor.
        :rtype: torch.Tensor
        """
        losses_norm = {}
        for condition in losses:
            losses[condition].backward(retain_graph=True)
            grads = []
            for param in self.model.parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            losses_norm[condition] = torch.norm(grads)
        self.weights = {
            condition: self.alpha
            * self.weights.get(condition, self.default_value_weights)
            + (1 - self.alpha)
            * losses_norm[condition]
            / sum(losses_norm.values())
            for condition in losses
        }
        return sum(
            self.weights[condition] * loss for condition, loss in losses.items()
        )
