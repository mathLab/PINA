import torch
from torch_geometric.nn import MessagePassing


class GraphIntegralLayer(MessagePassing):
    """
    TODO: Add documentation
    """
    def __init__(
            self,
            width,
            edges_features,
            n_layers=2,
            layers=None,
            inner_size=None,
            internal_func=None,
            external_func=None
    ):
        """
        Initialize the Graph Integral Layer, inheriting from the MessagePassing class of PyTorch Geometric.

        :param width: The width of the hidden representation of the nodes features
        :type width: int
        :param edges_features: The number of edge features.
        :type edges_features: int
        :param n_layers: The number of layers in the Feed Forward Neural Network used to compute the representation of the edges features.
        :type n_layers: int
        """
        from pina.model import FeedForward
        super(GraphIntegralLayer, self).__init__(aggr='mean')
        self.width = width
        if layers is None and inner_size is None:
            inner_size = width
        self.dense = FeedForward(input_dimensions=edges_features,
                                 output_dimensions=width ** 2,
                                 n_layers=n_layers,
                                 layers=layers,
                                 inner_size=inner_size,
                                 func=internal_func)
        self.W = torch.nn.Linear(width, width)
        self.func = external_func()

    def message(self, x_j, edge_attr):
        """
        This function computes the message passed between the nodes of the graph. Overwrite the default message function defined in the MessagePassing class.

        :param x_j: The node features of the neighboring.
        :type x_j: torch.Tensor
        :param edge_attr: The edge features.
        :type edge_attr: torch.Tensor
        :return: The message passed between the nodes of the graph.
        :rtype: torch.Tensor
        """
        x = self.dense(edge_attr).view(-1, self.width, self.width)
        return torch.einsum('bij,bj->bi', x, x_j)

    def update(self, aggr_out, x):
        """
        This function updates the node features of the graph. Overwrite the default update function defined in the MessagePassing class.

        :param aggr_out: The aggregated messages.
        :type aggr_out: torch.Tensor
        :param x: The node features.
        :type x: torch.Tensor
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        aggr_out = aggr_out + self.W(x)
        return aggr_out

    def forward(self, x, edge_index, edge_attr):
        """
        The forward pass of the Graph Integral Layer.

        :param x: Node features.
        :type x: torch.Tensor
        :param edge_index: Edge index.
        :type edge_index: torch.Tensor
        :param edge_attr: Edge features.
        :type edge_attr: torch.Tensor
        :return: Output of a single iteration over the Graph Integral Layer.
        :rtype: torch.Tensor
        """
        return self.func(
            self.propagate(edge_index, x=x, edge_attr=edge_attr)
        )
