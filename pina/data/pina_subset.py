"""
Module for PinaSubset class
"""
from pina import LabelTensor
from torch import Tensor


class PinaSubset:
    """
    TODO
    """
    __slots__ = ['dataset', 'indices']

    def __init__(self, dataset, indices):
        """
        TODO
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        """
        TODO
        """
        return len(self.indices)

    def __getattr__(self, name):
        tensor = self.dataset.__getattribute__(name)
        if isinstance(tensor, (LabelTensor, Tensor)):
            return tensor[self.indices]
        if isinstance(tensor, list):
            return [tensor[i] for i in self.indices]
        raise AttributeError("No attribute named {}".format(name))
