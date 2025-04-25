"""Module for the Deep Tensor Network block."""

import torch
from torch_geometric.nn import MessagePassing
from ....utils import check_consistency


class DeepTensorNetworkBlock(MessagePassing):
    """
    Implementation of the Deep Tensor Network block.

    This block is used to perform message-passing between nodes and edges in a
    graph neural network, following the scheme proposed by Schutt et al. (2017).
    It serves as an inner block in a larger graph neural network architecture.

    The message between two nodes connected by an edge is computed by applying a
    linear transformation to the sender node features and the edge features,
    followed by a non-linear activation function. Messages are then aggregated
    using an aggregation scheme (e.g., sum, mean, min, max, or product).

    The update step is performed by a simple addition of the incoming messages
    to the node features.

    .. seealso::

        **Original reference**: Schutt, K., Arbabzadah, F., Chmiela, S. et al.
        *Quantum-Chemical Insights from Deep Tensor Neural Networks*.
        Nature Communications 8, 13890 (2017).
        DOI: `<https://doi.org/10.1038/ncomms13890>_`.
    """

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        activation=torch.nn.Tanh,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        """
        Initialization of the :class:`DeepTensorNetworkBlocklock` class.

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
        :raises ValueError: If `edge_feature_dim` is not a positive integer.
        """
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)

        # Check consistency
        check_consistency(node_feature_dim, int)
        check_consistency(edge_feature_dim, int)

        # Check values
        if node_feature_dim <= 0:
            raise ValueError(
                "`node_feature_dim` must be a positive integer,"
                f" got {node_feature_dim}."
            )

        if edge_feature_dim <= 0:
            raise ValueError(
                "`edge_feature_dim` must be a positive integer,"
                f" got {edge_feature_dim}."
            )

        # Initialize parameters
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.activation = activation

        # Layer for processing node features
        self.node_layer = torch.nn.Linear(
            in_features=self.node_feature_dim,
            out_features=self.node_feature_dim,
            bias=True,
        )

        # Layer for processing edge features
        self.edge_layer = torch.nn.Linear(
            in_features=self.edge_feature_dim,
            out_features=self.node_feature_dim,
            bias=True,
        )

        # Layer for computing the message
        self.message_layer = torch.nn.Linear(
            in_features=self.node_feature_dim,
            out_features=self.node_feature_dim,
            bias=False,
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the block, triggering the message-passing routine.

        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge indeces.
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor | LabelTensor
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """
        Compute the message to be passed between nodes and edges.

        :param x_j: The node features of the sender nodes.
        :type x_j: torch.Tensor | LabelTensor
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor | LabelTensor
        :return: The message to be passed.
        :rtype: torch.Tensor
        """
        # Process node and edge features
        filter_node = self.node_layer(x_j)
        filter_edge = self.edge_layer(edge_attr)

        # Compute the message to be passed
        message = self.message_layer(filter_node * filter_edge)

        return self.activation(message)

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
