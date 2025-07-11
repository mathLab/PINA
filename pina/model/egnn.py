import torch
import torch.nn as nn

class EquivariantGraphNeuralNetwork(nn.Module):
    def __init__(self,
        node_feature_dim,
        edge_feature_dim,
        pos_dim,
        n_egnn_layers = 2,
        has_velo=False,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        activation=torch.nn.SiLU,
        aggr="add",
        node_dim=-2,
        flow="source_to_target"
    ):
        super().__init__()

        self.layers - nn.ModuleList()
        self.n_egnn_layers = n_egnn_layers
        self.has_velo = has_velo

        


