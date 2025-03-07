"""Module for PINA Torch Optimizer"""

import torch

from ..utils import check_consistency
from .optimizer_interface import Optimizer


class TorchOptimizer(Optimizer):
    """
    TODO

    :param Optimizer: _description_
    :type Optimizer: _type_
    """

    def __init__(self, optimizer_class, **kwargs):
        """
        TODO

        :param optimizer_class: _description_
        :type optimizer_class: _type_
        """
        check_consistency(optimizer_class, torch.optim.Optimizer, subclass=True)

        self.optimizer_class = optimizer_class
        self.kwargs = kwargs
        self._optimizer_instance = None

    def hook(self, parameters):
        """
        TODO

        :param parameters: _description_
        :type parameters: _type_
        """
        self._optimizer_instance = self.optimizer_class(
            parameters, **self.kwargs
        )

    @property
    def instance(self):
        """
        Optimizer instance.
        """
        return self._optimizer_instance
