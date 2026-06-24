"""Training callbacks for PINA lifecycle management.

This module provides specialized callbacks for training Scientific Machine
Learning models, including adaptive sample refinement (R3), optimizer
switching logic, and data normalization utilities.

:Example:

    >>> from pina.callback import MetricTracker, PINAProgressBar
    >>> tracker = MetricTracker()
    >>> # bar = PINAProgressBar()
"""

__all__ = [
    "SwitchOptimizer",
    "SwitchScheduler",
    "DataNormalizer",
    "PINAProgressBar",
    "MetricTracker",
    "RefinementInterface",
    "BaseRefinement",
    "R3Refinement",
]

from pina._src.callback.processing.pina_progress_bar import PINAProgressBar
from pina._src.callback.processing.metric_tracker import MetricTracker
from pina._src.callback.processing.data_normalizer import DataNormalizer
from pina._src.callback.optim.switch_optimizer import SwitchOptimizer
from pina._src.callback.optim.switch_scheduler import SwitchScheduler
from pina._src.callback.refinement.base_refinement import BaseRefinement
from pina._src.callback.refinement.r3_refinement import R3Refinement
from pina._src.callback.refinement.refinement_interface import (
    RefinementInterface,
)

# Back-compatibility with version 0.2, to be removed soon
import warnings

_DEPRECATED_IMPORTS = {"NormalizerDataCallback": "DataNormalizer"}


def __getattr__(name):
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.callback' is deprecated; use "
            f"pina.callback.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]
