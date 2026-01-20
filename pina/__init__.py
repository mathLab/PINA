"""
PINA: Physics-Informed Neural Analysis.

A specialized framework for Scientific Machine Learning (SciML), providing
tools for Physics-Informed Neural Networks (PINNs), Neural Operators,
and data-driven physical modeling.
"""

__all__ = [
    "LabelTensor",
    "Trainer",
    "Condition",
    "PinaDataModule",
    "Graph",
    "SolverInterface",
    "MultiSolverInterface",
]

from pina._src.core.label_tensor import LabelTensor
from pina._src.core.graph import Graph
from pina._src.solver.solver import SolverInterface, MultiSolverInterface
from pina._src.core.trainer import Trainer
from pina._src.condition.condition import Condition
from pina._src.data.data_module import PinaDataModule
