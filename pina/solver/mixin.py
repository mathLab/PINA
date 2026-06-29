"""Module for solver mixins.

"""

__all__ = [
    "SingleModelMixin",
    "MultiModelMixin",
    "EnsembleMixin",
    "ManualOptimizationMixin",
    "ConditionAggregatorMixin",
    "AutoregressiveMixin",
    "PhysicsInformedMixin",
    "ResidualBasedAttentionMixin",
    "GradientEnhancedMixin",
]

from pina._src.solver.mixin.single_model_mixin import SingleModelMixin
from pina._src.solver.mixin.multi_model_mixin import MultiModelMixin
from pina._src.solver.mixin.ensemble_mixin import EnsembleMixin
from pina._src.solver.mixin.manual_optimization_mixin import (
    ManualOptimizationMixin,
)
from pina._src.solver.mixin.condition_aggregator_mixin import (
    ConditionAggregatorMixin,
)
from pina._src.solver.mixin.autoregressive_mixin import AutoregressiveMixin
from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.solver.mixin.residual_based_attention_mixin import (
    ResidualBasedAttentionMixin,
)
from pina._src.solver.mixin.gradient_enhanced_mixin import GradientEnhancedMixin
