"""Module containing utilities for dataset and data loader management."""

__all__ = [
    "DataModule",
    "_SingleBatchDataLoader",
    "_Aggregator",
    "_Creator",
    "_ConditionSubset",
]

from pina._src.data.data_module import DataModule
from pina._src.data.single_batch_data_loader import _SingleBatchDataLoader
from pina._src.data.aggregator import _Aggregator
from pina._src.data.creator import _Creator
from pina._src.data.condition_subset import _ConditionSubset


# Back-compatibility with version 0.2, to be removed soon
import warnings

_DEPRECATED_IMPORTS = {"PinaDataModule": "DataModule"}


def __getattr__(name):
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.data' is deprecated; use "
            f"pina.data.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]
