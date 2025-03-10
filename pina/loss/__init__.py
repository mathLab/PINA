"""
Module for loss functions and weighting functions.
"""

__all__ = [
    "LossInterface",
    "LpLoss",
    "PowerLoss",
    "WeightingInterface",
    "ScalarWeighting",
]

from .loss_interface import LossInterface
from .power_loss import PowerLoss
from .lp_loss import LpLoss
from .weighting_interface import WeightingInterface
from .scalar_weighting import ScalarWeighting
