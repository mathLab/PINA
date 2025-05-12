"""Module for the Radial Field Network block."""

import torch
from ....model import FeedForward
from torch_geometric.nn import MessagePassing
from ....utils import check_consistency


class RadialFieldNetworkBlock(MessagePassing):
    """
    Implementation of the Radial Field Network block.

    This block is used to perform message-passing between nodes and edges in a
    graph neural network, following the scheme proposed by Köhler et al. (2020).
    It serves as an inner block in a larger graph neural network architecture.

    The message between two nodes connected by an edge is computed by applying a
    linear transformation to the sender node features and the edge features,
    followed by a non-linear activation function. Messages are then aggregated
    using an aggregation scheme (e.g., sum, mean, min, max, or product).

    The update step is performed by a simple addition of the incoming messages
    to the node features.

    .. seealso::

        **Original reference** Köhler, J., Klein, L., & Noé, F. (2020, November). 
        Equivariant flows: exact likelihood generative learning for symmetric densities. 
        In International conference on machine learning (pp. 5361-5370). PMLR.
    """

        

    def __init__(
        self,
        node_feature_dim,
        hidden_dim,
        radial_hidden_dim=16,
        n_radial_layers=2,
        activation=torch.nn.ReLU,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        """
        Initialization of the :class:`RadialFieldNetworkBlock` class.

        :param int node_feature_dim: The dimension of the node features.
        :param int edge_feature_dim: The dimension of the edge features.
        :param torch.nn.Module activation: The activation function.
            Default is :class:`torch.nn.Tanh`.
        :param str aggr: The aggregation scheme to use for message passing.
            Available options are "add", "mean", "min", "max", "mul".
            See :class:`torch_geometric.nn.MessagePassing` for more details.
            Default is "add".
        :param int node_dim: The axis along which to propagate. Default is -2.
        :param str flow: The direction of message passing. Available options
            are "source_to_target" and "target_to_source".
            The "source_to_target" flow means that messages are sent from
            the source node to the target node, while the "target_to_source"
            flow means that messages are sent from the target node to the
            source node. See :class:`torch_geometric.nn.MessagePassing` for more
            details. Default is "source_to_target".
        :raises ValueError: If `node_feature_dim` is not a positive integer.
        """
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)

        # Check consistency
        check_consistency(node_feature_dim, int)

        # Check values
        if node_feature_dim <= 0:
            raise ValueError(
                "`node_feature_dim` must be a positive integer,"
                f" got {node_feature_dim}."
            )

        # Initialize parameters
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Layer for processing node features
        self.radial_field = FeedForward(
            input_dimensions=1,
            output_dimensions=1,
            inner_size=radial_hidden_dim,
            n_layers=n_radial_layers,
            func=self.activation,
        )


    def forward(self, x, edge_index):
        """
        Forward pass of the block, triggering the message-passing routine.

        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge indices. In the original formulation, 
        the messages are aggregated from all nodes, not only from the neighbours. 
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        return self.propagate(edge_index=edge_index, x=x)

    def message(self, x_j, x_i):
        """
        Compute the message to be passed between nodes and edges.

        :param x_j: Node features of the sender nodes.
        :type x_j: torch.Tensor | LabelTensor
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor | LabelTensor
        :return: The message to be passed.
        :rtype: torch.Tensor
        """
        r = torch.norm(x_i-x_j)
        

        return self.radial_field(r)*(x_i-x_j)


    def update(self, message, x):
        """
        Update the node features with the received messages.

        :param torch.Tensor message: The message to be passed.
        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        return x + message
