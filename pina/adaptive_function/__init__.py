"""Adaptive activation functions with learnable parameters.

This module provides implementations of standard activation functions (ReLU,
SiLU, Tanh, etc.) augmented with trainable weights, as well as specialized
functions like SIREN, designed to improve convergence in PINNs and Neural
Operators.
"""

__all__ = [
    "AdaptiveFunctionInterface",
    "BaseAdaptiveFunction",
    "AdaptiveCELU",
    "AdaptiveELU",
    "AdaptiveExp",
    "AdaptiveGELU",
    "AdaptiveMish",
    "AdaptiveReLU",
    "AdaptiveSigmoid",
    "AdaptiveSiLU",
    "AdaptiveSIREN",
    "AdaptiveSoftmax",
    "AdaptiveSoftmin",
    "AdaptiveTanh",
]

from pina._src.adaptive_function.adaptive_function_interface import (
    AdaptiveFunctionInterface,
)
from pina._src.adaptive_function.base_adaptive_function import (
    BaseAdaptiveFunction,
)
from pina._src.adaptive_function.adaptive_celu import AdaptiveCELU
from pina._src.adaptive_function.adaptive_elu import AdaptiveELU
from pina._src.adaptive_function.adaptive_exp import AdaptiveExp
from pina._src.adaptive_function.adaptive_gelu import AdaptiveGELU
from pina._src.adaptive_function.adaptive_mish import AdaptiveMish
from pina._src.adaptive_function.adaptive_relu import AdaptiveReLU
from pina._src.adaptive_function.adaptive_sigmoid import AdaptiveSigmoid
from pina._src.adaptive_function.adaptive_silu import AdaptiveSiLU
from pina._src.adaptive_function.adaptive_siren import AdaptiveSIREN
from pina._src.adaptive_function.adaptive_softmax import AdaptiveSoftmax
from pina._src.adaptive_function.adaptive_softmin import AdaptiveSoftmin
from pina._src.adaptive_function.adaptive_tanh import AdaptiveTanh
