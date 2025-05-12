"""Module for the Schnet block."""

import torch
from ....model import FeedForward
from torch_geometric.nn import MessagePassing
from ....utils import check_consistency


class SchnetBlock(MessagePassing):
    """
    Implementation of the Schnet block.

    This block is used to perform message-passing between nodes and edges in a
    graph neural network, following the scheme proposed by Schütt et al. (2017).
    It serves as an inner block in a larger graph neural network architecture.

    The message between two nodes connected by an edge is computed by applying a
    linear transformation to the sender node features and the edge features,
    followed by a non-linear activation function. Messages are then aggregated
    using an aggregation scheme (e.g., sum, mean, min, max, or product).

    The update step is performed by a simple addition of the incoming messages
    to the node features.

    .. seealso::

        **Original reference** Schütt, K., Kindermans, P. J., Sauceda Felix, H. E., Chmiela, S., Tkatchenko, A., & Müller, K. R. (2017). 
        Schnet: A continuous-filter convolutional neural network for modeling quantum interactions. 
        Advances in neural information processing systems, 30.
    """

        

    def __init__(
        self,
        node_feature_dim,
        node_pos_dim,
        hidden_dim,
        radial_hidden_dim=16,
        n_message_layers=2,
        n_update_layers=2,
        n_radial_layers=2,
        activation=torch.nn.ReLU,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        """
        Initialization of the :class:`SchnetBlock` class.

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

        # Check values
        if node_feature_dim <= 0:
            raise ValueError(
                "`node_feature_dim` must be a positive integer,"
                f" got {node_feature_dim}."
            )


        # Initialize parameters
        self.node_feature_dim = node_feature_dim
        self.node_pos_dim = node_pos_dim
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
        
        self.update_net = FeedForward(
            input_dimensions=self.node_pos_dim + self.hidden_dim,
            output_dimensions=self.hidden_dim,
            inner_size=self.hidden_dim,
            n_layers=n_update_layers,
            func=self.activation,
        )

        self.message_net = FeedForward(
            input_dimensions=self.node_feature_dim,
            output_dimensions=self.node_pos_dim + self.hidden_dim,
            inner_size=self.hidden_dim,
            n_layers=n_message_layers,
            func=self.activation,
        )


    def forward(self, x, pos, edge_index):
        """
        Forward pass of the block, triggering the message-passing routine.

        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge indices. In the original formulation, 
        the messages are aggregated from all nodes, not only from the neighbours. 
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        return self.propagate(edge_index=edge_index, x=x, pos=pos)

    def message(self, x_i, pos_i ,pos_j):
        """
        Compute the message to be passed between nodes and edges.

        :param x_j: Node features of the sender nodes.
        :type x_j: torch.Tensor | LabelTensor
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor | LabelTensor
        :return: The message to be passed.
        :rtype: torch.Tensor
        """  

        return self.radial_field(torch.norm(pos_i-pos_j))*self.message_net(x_i)


    def update(self, message, pos):
        """
        Update the node features with the received messages.

        :param torch.Tensor message: The message to be passed.
        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :return: The concatenation of the update position features and the updated node features.
        :rtype: torch.Tensor
        """
        return self.update_net(torch.cat((pos, message), dim=-1))
