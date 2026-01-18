"""Training callbacks for PINA lifecycle management.

This module provides specialized callbacks for training Scientific Machine
Learning models, including adaptive sample refinement (R3), optimizer
switching logic, and data normalization utilities.
"""

__all__ = [
    "SwitchOptimizer",
    "SwitchScheduler",
    "NormalizerDataCallback",
    "PINAProgressBar",
    "MetricTracker",
    "R3Refinement",
]

from pina._src.callback.optim.switch_optimizer import SwitchOptimizer
from pina._src.callback.optim.switch_scheduler import SwitchScheduler
from pina._src.callback.processing.normalizer_data_callback import NormalizerDataCallback
from pina._src.callback.processing.pina_progress_bar import PINAProgressBar
from pina._src.callback.processing.metric_tracker import MetricTracker
from pina._src.callback.refinement.r3_refinement import R3Refinement
