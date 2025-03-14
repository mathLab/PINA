"""Module for the Neural model classes."""

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
]

from .feed_forward import FeedForward, ResidualFeedForward
from .multi_feed_forward import MultiFeedForward
from .deeponet import DeepONet, MIONet
from .fourier_neural_operator import FNO, FourierIntegralKernel
from .kernel_neural_operator import KernelNeuralOperator
from .average_neural_operator import AveragingNeuralOperator
from .low_rank_neural_operator import LowRankNeuralOperator
from .spline import Spline
from .graph_neural_operator import GraphNeuralOperator
