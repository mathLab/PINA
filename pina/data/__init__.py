"""Data management utilities for PINA.

This module provides specialized Dataset and DataModule implementations
designed to handle physical coordinates, experimental observations, and
graph-structured data within the PINA training pipeline.
"""

from pina._src.data.data_module import (
    PinaDataModule,
)

__all__ = [
    "PinaDataModule",
]
