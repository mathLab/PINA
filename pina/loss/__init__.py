"""Loss functions and balancing strategies for multi-objective optimization.

This module provides standard error metrics (Lp, Power loss) and sophisticated
weighting schemes designed to balance residual, boundary, and data-driven loss
terms, including dynamic methods like Neural Tangent Kernel (NTK) and
self-adaptive weighting.
"""

__all__ = [
    "LossInterface",
    "LpLoss",
    "PowerLoss",
    "WeightingInterface",
    "ScalarWeighting",
    "NeuralTangentKernelWeighting",
    "SelfAdaptiveWeighting",
    "LinearWeighting",
]

from pina._src.loss.loss_interface import LossInterface
from pina._src.loss.power_loss import PowerLoss
from pina._src.loss.lp_loss import LpLoss
from pina._src.loss.weighting_interface import WeightingInterface
from pina._src.loss.scalar_weighting import ScalarWeighting
from pina._src.loss.ntk_weighting import NeuralTangentKernelWeighting
from pina._src.loss.self_adaptive_weighting import SelfAdaptiveWeighting
from pina._src.loss.linear_weighting import LinearWeighting
