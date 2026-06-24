"""Module for the message passing blocks of the graph neural models.

:Example:

    >>> from pina.model.block.message_passing import (
    ...     InteractionNetworkBlock,
    ...     DeepTensorNetworkBlock,
    ... )
    >>> import torch
    >>> block = InteractionNetworkBlock(input_dim=3, hidden_dim=16)
    >>> # block processes graph-structured data
"""

__all__ = [
    "InteractionNetworkBlock",
    "DeepTensorNetworkBlock",
    "EnEquivariantNetworkBlock",
    "RadialFieldNetworkBlock",
    "EquivariantGraphNeuralOperatorBlock",
]

from pina._src.model.block.message_passing.interaction_network_block import (
    InteractionNetworkBlock,
)
from pina._src.model.block.message_passing.deep_tensor_network_block import (
    DeepTensorNetworkBlock,
)
from pina._src.model.block.message_passing.en_equivariant_network_block import (
    EnEquivariantNetworkBlock,
)
from pina._src.model.block.message_passing.radial_field_network_block import (
    RadialFieldNetworkBlock,
)
from pina._src.model.block.message_passing.equivariant_graph_neural_operator_block import (
    EquivariantGraphNeuralOperatorBlock,
)
