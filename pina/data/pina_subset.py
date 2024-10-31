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

    def __init__(self, dataset, indices, require_grad=False):
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
        if isinstance(self.indices, slice):
            return self.indices.stop - self.indices.start
        return len(self.indices)

    def __getattr__(self, name):
        tensor = self.dataset.__getattribute__(name)
        if isinstance(tensor, (LabelTensor, Tensor)):
            if isinstance(self.indices, slice):
                tensor = tensor[self.indices]
                if (tensor.device != self.dataset.device
                        and tensor.dtype == float32):
                    tensor = tensor.to(self.dataset.device)
            elif isinstance(self.indices, list):
                tensor = tensor[[self.indices]].to(self.dataset.device)
            else:
                raise ValueError(f"Indices type {type(self.indices)} not "
                                 f"supported")
            return tensor.requires_grad_(
                self.require_grad) if tensor.dtype == float32 else tensor
        if isinstance(tensor, list):
            if isinstance(self.indices, list):
                return [tensor[i] for i in self.indices]
            return tensor[self.indices]
        raise AttributeError(f"No attribute named {name}")
