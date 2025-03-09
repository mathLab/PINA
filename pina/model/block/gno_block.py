"""
Module containing the Graph Integral Layer class.
"""

import torch
from torch_geometric.nn import MessagePassing


class GNOBlock(MessagePassing):
    """
    Graph Neural Operator (GNO) Block using PyG MessagePassing.
    """

    def __init__(
        self,
        width,
        edges_features,
        n_layers=2,
        layers=None,
        inner_size=None,
        internal_func=None,
        external_func=None,
    ):
        """
        Initialize the GNOBlock.

        :param width: Hidden dimension of node features.
        :param edges_features: Number of edge features.
        :param n_layers: Number of layers in edge transformation MLP.
        """
        # Avoid circular import. I need to import FeedForward here
        # to avoid circular import with FeedForward itself.
        # pylint: disable=import-outside-toplevel
        from ...model.feed_forward import FeedForward

        super().__init__(aggr="mean")  # Uses PyG's default aggregation
        self.width = width

        if layers is None and inner_size is None:
            inner_size = width

        self.dense = FeedForward(
            input_dimensions=edges_features,
            output_dimensions=width**2,
            n_layers=n_layers,
            layers=layers,
            inner_size=inner_size,
            func=internal_func,
        )

        self.W = torch.nn.Linear(width, width)
        self.func = external_func()

    def message_and_aggregate(self, edge_index, x, edge_attr):
        """
        Combines message and aggregation.

        :param edge_index: COO format edge indices.
        :param x: Node feature matrix [num_nodes, width].
        :param edge_attr: Edge features [num_edges, edge_dim].
        :return: Aggregated messages.
        """
        # Edge features are transformed into a matrix of shape
        # [num_edges, width, width]
        x_ = self.dense(edge_attr).view(-1, self.width, self.width)
        # Messages are computed as the product of the edge features
        messages = torch.einsum("bij,bj->bi", x_, x[edge_index[0]])
        # Aggregation is performed using the mean (set in the constructor)
        return self.aggregate(messages, edge_index[1])

    def edge_update(self, edge_attr):
        """
        Updates edge features.
        """
        return edge_attr

    def update(self, aggr_out, x):
        """
        Updates node features.

        :param aggr_out: Aggregated messages.
        :param x: Node feature matrix.
        :return: Updated node features.
        """
        return aggr_out + self.W(x)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GNOBlock.

        :param x: Node features.
        :param edge_index: Edge indices.
        :param edge_attr: Edge features.
        :return: Updated node features.
        """
        return self.func(self.propagate(edge_index, x=x, edge_attr=edge_attr))
