"""Module for the Time Weighting."""

import torch
from .time_weighting_interface import TimeWeightingInterface


class ConstantTimeWeighting(TimeWeightingInterface):
    """
    Weighting scheme that assigns equal weight to all time steps.
    """

    def __call__(self, num_steps, device):
        return torch.ones(num_steps, device=device) / num_steps


class ExponentialTimeWeighting(TimeWeightingInterface):
    """
    Weighting scheme change exponentially with time.
    gamma > 1.0: increasing weights
    0 < gamma < 1.0: decreasing weights
    weight at time t is gamma^t
    """

    def __init__(self, gamma=0.9):
        """
        Initialization of the :class:`ExponentialTimeWeighting` class.
        :param float gamma: The decay factor. Default is 0.9.
        """
        self.gamma = gamma

    def __call__(self, num_steps, device):
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        weights = self.gamma**steps
        return weights / weights.sum()


class LinearTimeWeighting(TimeWeightingInterface):
    """
    Weighting scheme that changes linearly from a start weight to an end weight.
    """

    def __init__(self, start=0.1, end=1.0):
        """
        Initialization of the :class:`LinearDecayTimeWeighting` class.

        :param float start: The starting weight. Default is 0.1.
        :param float end: The ending weight. Default is 1.0.
        """
        self.start = start
        self.end = end

    def __call__(self, num_steps, device):
        if num_steps == 1:
            return torch.ones(1, device=device)

        weights = torch.linspace(self.start, self.end, num_steps, device=device)
        return weights / weights.sum()
