"""Module for Location class."""

from .location import Location
from ..label_tensor import LabelTensor


class Difference(Location):
    """
    """

    def __init__(self, first, second):

        self.first = first
        self.second = second

    def sample(self, n, mode='random', variables='all'):
        """
        """
        assert mode == 'random', 'Only random mode is implemented'

        samples = []
        while len(samples) < n:
            sample = self.first.sample(1, 'random')
            if not self.second.is_inside(sample):
                samples.append(sample)

        import torch
        return LabelTensor(torch.tensor(samples), labels=['x', 'y'])
