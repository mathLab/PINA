"""
Module for PinaSubset class
"""
from pina import LabelTensor
from torch import Tensor, float32


class PinaSubset:
    """
    TODO
    """
    __slots__ = ['dataset', 'indices', 'require_grad']

    def __init__(self, dataset, indices, require_grad=True):
        """
        TODO
        """
        self.dataset = dataset
        self.indices = indices
        self.require_grad = require_grad

    def __len__(self):
        """
        TODO
        """
        return len(self.indices)

    def __getattr__(self, name):
        tensor = self.dataset.__getattribute__(name)
        if isinstance(tensor, (LabelTensor, Tensor)):
            tensor = tensor[[self.indices]].to(self.dataset.device)
            return tensor.requires_grad_(
                self.require_grad) if tensor.dtype == float32 else tensor
        if isinstance(tensor, list):
            return [tensor[i] for i in self.indices]
        raise AttributeError(f"No attribute named {name}")
