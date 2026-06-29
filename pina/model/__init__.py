"""Module for the Neural model classes.

"""

__all__ = [
    "FeedForward",
    "ResidualFeedForward",
    "MultiFeedForward",
    "DeepONet",
    "MIONet",
    "FNO",
    "FourierIntegralKernel",
    "KernelNeuralOperator",
    "AveragingNeuralOperator",
    "LowRankNeuralOperator",
    "Spline",
    "GraphNeuralOperator",
    "PirateNet",
    "EquivariantGraphNeuralOperator",
    "SINDy",
    "SplineSurface",
    "VectorizedSpline",
    "KolmogorovArnoldNetwork",
]

from pina._src.model.feed_forward import FeedForward, ResidualFeedForward
from pina._src.model.multi_feed_forward import MultiFeedForward
from pina._src.model.deeponet import DeepONet, MIONet
from pina._src.model.fourier_neural_operator import FNO, FourierIntegralKernel
from pina._src.model.kernel_neural_operator import KernelNeuralOperator
from pina._src.model.average_neural_operator import AveragingNeuralOperator
from pina._src.model.low_rank_neural_operator import LowRankNeuralOperator
from pina._src.model.spline import Spline
from pina._src.model.spline_surface import SplineSurface
from pina._src.model.graph_neural_operator import GraphNeuralOperator
from pina._src.model.pirate_network import PirateNet
from pina._src.model.equivariant_graph_neural_operator import (
    EquivariantGraphNeuralOperator,
)
from pina._src.model.sindy import SINDy
from pina._src.model.vectorized_spline import VectorizedSpline
from pina._src.model.kolmogorov_arnold_network import KolmogorovArnoldNetwork
