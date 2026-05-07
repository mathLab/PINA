"""Module for loss functions."""

__all__ = [
    "DualLossInterface",
    "BaseLoss",
    "LpLoss",
    "PowerLoss",
]

from pina._src.loss.loss_interface import DualLossInterface
from pina._src.loss.base_loss import BaseLoss
from pina._src.loss.power_loss import PowerLoss
from pina._src.loss.lp_loss import LpLoss

# Back-compatibility with version 0.2, to be removed soon
import warnings
import importlib

_DEPRECATED_IMPORTS = {
    "WeightingInterface": "pina.weighting",
    "ScalarWeighting": "pina.weighting",
    "NeuralTangentKernelWeighting": "pina.weighting",
    "SelfAdaptiveWeighting": "pina.weighting",
    "LinearWeighting": "pina.weighting",
}


def __getattr__(name):
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.loss' is deprecated; "
            f"use 'pina.weighting' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        module = importlib.import_module(_DEPRECATED_IMPORTS[name], __name__)
        return getattr(module, name)
