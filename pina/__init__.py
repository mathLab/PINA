"""
A specialized framework for Scientific Machine Learning (SciML), providing
tools for Physics-Informed Neural Networks (PINNs), Neural Operators,
and data-driven physical modelling.

:Example:

    >>> from pina import LabelTensor
    >>> import torch
    >>> t = LabelTensor(torch.rand(3, 2), ['x', 'y'])
    >>> t.labels
    ['x', 'y']
"""

__all__ = [
    "LabelTensor",
    "Trainer",
    "Condition",
    "DataModule",
    "Graph",
]

from pina._src.core.label_tensor import LabelTensor
from pina._src.core.graph import Graph
from pina._src.core.trainer import Trainer
from pina._src.condition.condition import Condition
from pina._src.data.data_module import DataModule

# Back-compatibility with version 0.2, to be removed soon
import warnings

_DEPRECATED_IMPORTS = {"PinaDataModule": "DataModule"}


def __getattr__(name):
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina' is deprecated; use "
            f"pina.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]
