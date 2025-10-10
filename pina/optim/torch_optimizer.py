"""Module for the PINA Torch Optimizer"""

import torch

from ..utils import check_consistency
from .core.optimizer_connector import OptimizerConnector


class TorchOptimizer(OptimizerConnector):
    """
    A wrapper class for using PyTorch optimizers.
    """

    def __init__(self, optimizer_class, **optimizer_class_kwargs):
        """
        Initialization of the :class:`TorchOptimizer` class.

        :param torch.optim.Optimizer optimizer_class: A
            :class:`torch.optim.Optimizer` class.
        :param dict kwargs: Additional parameters passed to ``optimizer_class``,
            see more
            `here <https://pytorch.org/docs/stable/optim.html#algorithms>`_.
        """
        # external checks
        check_consistency(optimizer_class, torch.optim.Optimizer, subclass=True)
        check_consistency(optimizer_class_kwargs, dict)
        super().__init__(optimizer_class, **optimizer_class_kwargs)
