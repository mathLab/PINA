import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from ...utils import check_positive_integer
from ...model import FeedForward


class EquivariantGraphNeuralOperator(MessagePassing):
    #! What is hidden_dim, what is node_dim=-2
    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        pos_dim,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        activation=torch.nn.SiLU,
        aggr="add",
        node_dim=-2,
        flow="source_to_target",
    ):
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)

        # Check values
        check_positive_integer(node_feature_dim, strict=True)
        check_positive_integer(edge_feature_dim, strict=False)
        check_positive_integer(pos_dim, strict=True)
        check_positive_integer(hidden_dim, strict=True)
        check_positive_integer(n_message_layers, strict=True)
        check_positive_integer(n_update_layers, strict=True)

        # Layer for computing the message
        

        # Layer for updating the node features
        

        # Layer for updating the node positions
        

    def forward(self):
        pass

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        diff = pos_i - pos_j

    def aggregate(self):
        pass

    def update(self):
        pass
