"""Module for the message passing blocks of the graph neural models."""

__all__ = [
    "InteractionNetworkBlock",
    "DeepTensorNetworkBlock",
]

from .interaction_network_block import InteractionNetworkBlock
from .deep_tensor_network_block import DeepTensorNetworkBlock
