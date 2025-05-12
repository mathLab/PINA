"""Module for the E(n) Equivariant Graph Neural Network block."""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class EnEquivariantGraphBlock(MessagePassing):
    """
    Implementation of the E(n) Equivariant Graph Neural Network block.

    This block is used to perform message-passing between nodes and edges in a
    graph neural network, following the scheme proposed by Satorras et al. (2021).
    It serves as an inner block in a larger graph neural network architecture.

    The message between two nodes connected by an edge is computed by applying a
    linear transformation to the sender node features and the edge features,
    followed by a non-linear activation function. Messages are then aggregated
    using an aggregation scheme (e.g., sum, mean, min, max, or product).

    The update step is performed by a simple addition of the incoming messages
    to the node features.

    .. seealso::

        **Original reference** Satorras, V. G., Hoogeboom, E., & Welling, M. (2021, July). 
        E (n) equivariant graph neural networks. 
        In International conference on machine learning (pp. 9323-9332). PMLR.
    """

    def __init__(
        self,
        channels_x,
        channels_m,
        channels_a,
        aggr: str = "add",
        hidden_channels: int = 64,
        **kwargs,
    ):
        """
        Initialization of the :class:`EnEquivariantGraphBlock` class.

        :param int channels_x: The dimension of the node features.
        :param int channels_m: The dimension of the Euclidean coordinates (should be =3).
        :param int channels_a: The dimension of the edge features.
        :param str aggr: The aggregation scheme to use for message passing.
            Available options are "add", "mean", "min", "max", "mul".
            See :class:`torch_geometric.nn.MessagePassing` for more details.
            Default is "add".
        :param int hidden_channels_dim: The hidden dimension in each MLPs initialized in the block.
        """
        super().__init__(aggr=aggr, **kwargs)

        self.phi_e = torch.nn.Sequential(
            torch.nn.Linear(2 * channels_x + 1 + channels_a, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, channels_m),
            torch.nn.LayerNorm(channels_m),
            torch.nn.SiLU(),
        )
        self.phi_pos = torch.nn.Sequential(
            torch.nn.Linear(channels_m, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, 1),
        )
        self.phi_x = torch.nn.Sequential(
            torch.nn.Linear(channels_x + channels_m, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, channels_x),
        )

    def forward(self, x, pos, edge_attr, edge_index, c=None):
        """
        Forward pass of the block, triggering the message-passing routine.

        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :param pos_i: 3D Euclidean coordinates.
        :type pos_i: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge indices. In the original formulation, 
        the messages are aggregated from all nodes, not only from the neighbours. 
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        if c is None:
            c = degree(edge_index[0], pos.shape[0]).unsqueeze(-1)
        return self.propagate(
            edge_index=edge_index, x=x, pos=pos, edge_attr=edge_attr, c=c
        )

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        """
        Compute the message to be passed between nodes and edges.

        :param x_i: Node features of the sender nodes.
        :type x_i: torch.Tensor | LabelTensor
        :param pos_i: 3D Euclidean coordinates of the sender nodes.
        :type pos_i: torch.Tensor | LabelTensor
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor | LabelTensor
        :return: The message to be passed.
        :rtype: torch.Tensor
        """
        mpos_ij = self.phi_e(
            torch.cat(
                [
                    x_i,
                    x_j,
                    torch.norm(pos_i - pos_j, dim=-1, keepdim=True) ** 2,
                    edge_attr,
                ],
                dim=-1,
            )
        )
        mpos_ij = (pos_i - pos_j) * self.phi_pos(mpos_ij)
        return  mpos_ij

    def update(self, message, x, pos, c):
        """
        Update the node features with the received messages.

        :param torch.Tensor message: The message to be passed.
        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :param pos: The 3D Euclidean coordinates of the nodes.
        :type pos: torch.Tensor | LabelTensor
        :param c: the constant that divides the aggregated message (it should be (M-1), where M is the number of nodes)
        :type pos: torch.Tensor 
        :return: The concatenation of the update position features and the updated node features.
        :rtype: torch.Tensor
        """
        x = self.phi_x(torch.cat([x, message], dim=-1))
        pos = pos + (message / c)
        return pos, x
