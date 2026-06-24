"""Module for weighting strategies in multi-objective optimization.

:Example:

    >>> from pina.weighting import LinearWeighting, ScalarWeighting
    >>> weighting = LinearWeighting(weights=[0.3, 0.7])
"""

__all__ = [
    "WeightingInterface",
    "BaseWeighting",
    "LinearWeighting",
    "NeuralTangentKernelWeighting",
    "_NoWeighting",
    "ScalarWeighting",
    "SelfAdaptiveWeighting",
]

from pina._src.weighting.weighting_interface import WeightingInterface
from pina._src.weighting.base_weighting import BaseWeighting
from pina._src.weighting.linear_weighting import LinearWeighting
from pina._src.weighting.ntk_weighting import NeuralTangentKernelWeighting
from pina._src.weighting.no_weighting import _NoWeighting
from pina._src.weighting.scalar_weighting import ScalarWeighting
from pina._src.weighting.self_adaptive_weighting import SelfAdaptiveWeighting
