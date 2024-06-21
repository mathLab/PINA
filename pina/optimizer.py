""" Module for PINA Optimizer """

import torch
from .utils import check_consistency

class Optimizer: # TODO improve interface
    pass


class TorchOptimizer(Optimizer):

    def __init__(self, optimizer_class, **kwargs):
        check_consistency(optimizer_class, torch.optim.Optimizer, subclass=True)

        self.optimizer_class = optimizer_class
        self.kwargs = kwargs

    def hook(self, parameters):
        self.optimizer_instance = self.optimizer_class(
            parameters, **self.kwargs
        )
