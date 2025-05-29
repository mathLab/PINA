"""Module for the Schnet block."""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops
from ....utils import check_positive_integer
from ....model import FeedForward


class SchnetBlock(MessagePassing):
    """
    Implementation of the Schnet block.

    This block is used to perform message-passing between nodes and edges in a
    graph neural network, following the scheme proposed by Schütt et al. in
    2017. It serves as an inner block in a larger graph neural network
    architecture.

    The message between two nodes connected by an edge is computed as the
    product of the output of a MLP applied to the norm of the distance of the
    node positions, and of another MLP applied to the node features. Messages
    are then aggregated using an aggregation scheme (e.g., sum, mean, min, max,
    or product).

    The update step is performed by applying another MLP to the concatenation of
    the incoming messages and the node features.

    .. seealso::

        **Original reference** Schütt, K., Kindermans, P. J., Sauceda Felix,
        H. E., Chmiela, S., Tkatchenko, A., Müller, K. R. (2017).
        *Schnet: A continuous-filter convolutional neural network for modeling
        quantum interactions.*
        Advances in Neural Information Processing Systems, 30.
        DOI: `<https://doi.org/10.48550/arXiv.1706.08566>`_.
    """

    def __init__(
        self,
        node_feature_dim,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        n_radial_layers=2,
        activation=torch.nn.SiLU,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        """
        Initialization of the :class:`SchnetBlock` class.

        :param int node_feature_dim: The dimension of the node features.
        :param int hidden_dim: The dimension of the hidden features.
            Default is 64.
        :param int n_message_layers: The number of layers in the message
            network. Default is 2.
        :param int n_update_layers: The number of layers in the update network.
            Default is 2.
        :param int n_radial_layers: The number of layers in the radial field
            network. Default is 2.
        :param torch.nn.Module activation: The activation function.
            Default is :class:`torch.nn.SiLU`.
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
        :raises AssertionError: If `node_feature_dim` is not a positive integer.
        :raises AssertionError: If `hidden_dim` is not a positive integer.
        :raises AssertionError: If `n_message_layers` is not a positive integer.
        :raises AssertionError: If `n_update_layers` is not a positive integer.
        :raises AssertionError: If `n_radial_layers` is not a positive integer.
        """
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)

        # Check values
        check_positive_integer(node_feature_dim, strict=True)
        check_positive_integer(hidden_dim, strict=True)
        check_positive_integer(n_message_layers, strict=True)
        check_positive_integer(n_update_layers, strict=True)
        check_positive_integer(n_radial_layers, strict=True)

        # Layer for processing node distances
        self.radial_net = FeedForward(
            input_dimensions=1,
            output_dimensions=1,
            inner_size=hidden_dim,
            n_layers=n_radial_layers,
            func=activation,
        )

        # Layer for computing the message
        self.message_net = FeedForward(
            input_dimensions=node_feature_dim,
            output_dimensions=node_feature_dim,
            inner_size=hidden_dim,
            n_layers=n_message_layers,
            func=activation,
        )

        # Layer for updating the node features
        self.update_net = FeedForward(
            input_dimensions=2 * node_feature_dim,
            output_dimensions=node_feature_dim,
            inner_size=hidden_dim,
            n_layers=n_update_layers,
            func=activation,
        )

    def forward(self, x, pos, edge_index):
        """
        Forward pass of the block, triggering the message-passing routine.

        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :param torch.Tensor edge_index: The edge indices.
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index=edge_index, x=x, pos=pos)

    def message(self, x_i, pos_i, pos_j):
        """
        Compute the message to be passed between nodes and edges.

        :param x_i: Node features of the sender nodes.
        :type x_i: torch.Tensor | LabelTensor
        :param pos_i: The node coordinates of the recipient nodes.
        :type pos_i: torch.Tensor | LabelTensor
        :param pos_j: The node coordinates of the sender nodes.
        :type pos_j: torch.Tensor | LabelTensor
        :return: The message to be passed.
        :rtype: torch.Tensor
        """
        rad = self.radial_net(torch.norm(pos_i - pos_j, dim=-1, keepdim=True))
        msg = self.message_net(x_i)
        return rad * msg

    def update(self, message, x):
        """
        Update the node features with the received messages.

        :param torch.Tensor message: The message to be passed.
        :param x: The node features.
        :type x: torch.Tensor | LabelTensor
        :return: The updated node features.
        :rtype: torch.Tensor
        """
        return self.update_net(torch.cat((x, message), dim=-1))
