import torch
from torch.nn import ReLU
from torch_geometric.nn.conv import GCNConv


class GNO_layer(torch.nn.Module):
    def __init__(self,
                 hidden_dimension,
                 func=None):
        super.__init__()
        if func is None:
            self.func = ReLU()
        self.conv = GCNConv(
            in_channels=hidden_dimension,
            out_channels=hidden_dimension)

    def forward(self, x, edge_index, edge_weight=None):
        return self.func(self.conv(x, edge_index, edge_weight))



