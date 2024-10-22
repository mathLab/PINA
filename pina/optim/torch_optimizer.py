""" Module for PINA Torch Optimizer """

import torch

from ..utils import check_consistency
from .optimizer_interface import Optimizer


class TorchOptimizer(Optimizer):

    def __init__(self, optimizer_class, **kwargs):
        check_consistency(optimizer_class, torch.optim.Optimizer, subclass=True)

        self.optimizer_class = optimizer_class
        self.kwargs = kwargs

    def hook(self, parameters):
        self.optimizer_instance = self.optimizer_class(parameters,
                                                       **self.kwargs)
