"""Architectural primitives and building blocks.

This module provides a comprehensive collection of neural network components,
ranging from standard units (Residual, Enhanced Linear) to specialized layers
for Scientific Machine Learning, including Neural Operator blocks (FNO, GNO,
AVNO), spectral convolutions, and coordinate embeddings (Fourier Features).
"""

__all__ = [
    "ContinuousConvBlock",
    "ResidualBlock",
    "EnhancedLinear",
    "SpectralConvBlock1D",
    "SpectralConvBlock2D",
    "SpectralConvBlock3D",
    "FourierBlock1D",
    "FourierBlock2D",
    "FourierBlock3D",
    "PODBlock",
    "OrthogonalBlock",
    "PeriodicBoundaryEmbedding",
    "FourierFeatureEmbedding",
    "AVNOBlock",
    "LowRankBlock",
    "RBFBlock",
    "GNOBlock",
    "PirateNetBlock",
]

from pina._src.model.block.convolution_2d import ContinuousConvBlock
from pina._src.model.block.residual import ResidualBlock, EnhancedLinear
from pina._src.model.block.spectral import (
    SpectralConvBlock1D,
    SpectralConvBlock2D,
    SpectralConvBlock3D,
)
from pina._src.model.block.fourier_block import FourierBlock1D, FourierBlock2D, FourierBlock3D
from pina._src.model.block.pod_block import PODBlock
from pina._src.model.block.orthogonal import OrthogonalBlock
from pina._src.model.block.embedding import PeriodicBoundaryEmbedding, FourierFeatureEmbedding
from pina._src.model.block.average_neural_operator_block import AVNOBlock
from pina._src.model.block.low_rank_block import LowRankBlock
from pina._src.model.block.rbf_block import RBFBlock
from pina._src.model.block.gno_block import GNOBlock
from pina._src.model.block.pirate_network_block import PirateNetBlock
