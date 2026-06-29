"""
Utility functions for tensor manipulation and input validation.

This module provides helper functions to manage tensor operations and ensure
data consistency across the PINA framework, supporting robust input checking
and seamless data merging.
"""

from pina._src.core.utils import (
    merge_tensors,
    check_consistency,
    check_positive_integer,
)

__all__ = [
    "merge_tensors",
    "check_consistency",
    "check_positive_integer",
]
