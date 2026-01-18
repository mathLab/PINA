"""Adaptive activation functions with learnable parameters.

This module provides implementations of standard activation functions (ReLU,
SiLU, Tanh, etc.) augmented with trainable weights, as well as specialized
functions like SIREN, designed to improve convergence in PINNs and Neural
Operators.
"""

__all__ = [
    "AdaptiveActivationFunctionInterface",
    "AdaptiveReLU",
    "AdaptiveSigmoid",
    "AdaptiveTanh",
    "AdaptiveSiLU",
    "AdaptiveMish",
    "AdaptiveELU",
    "AdaptiveCELU",
    "AdaptiveGELU",
    "AdaptiveSoftmin",
    "AdaptiveSoftmax",
    "AdaptiveSIREN",
    "AdaptiveExp",
]

from pina._src.adaptive_function.adaptive_function import (
    AdaptiveReLU,
    AdaptiveSigmoid,
    AdaptiveTanh,
    AdaptiveSiLU,
    AdaptiveMish,
    AdaptiveELU,
    AdaptiveCELU,
    AdaptiveGELU,
    AdaptiveSoftmin,
    AdaptiveSoftmax,
    AdaptiveSIREN,
    AdaptiveExp,
)
from pina._src.adaptive_function.adaptive_function_interface import AdaptiveActivationFunctionInterface
