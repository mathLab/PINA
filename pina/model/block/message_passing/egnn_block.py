"""Module for the E(n) Equivariant Graph Neural Network block."""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class EnEquivariantGraphBlock(MessagePassing):
    """
    TODO
    """

    def __init__(
        self,
        channels_h,
        channels_m,
        channels_a,
        aggr: str = "add",
        hidden_channels: int = 64,
        **kwargs,
    ):
        """
        TODO
        """
        super().__init__(aggr=aggr, **kwargs)

        self.phi_e = torch.nn.Sequential(
            torch.nn.Linear(2 * channels_h + 1 + channels_a, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, channels_m),
            torch.nn.LayerNorm(channels_m),
            torch.nn.SiLU(),
        )
        self.phi_x = torch.nn.Sequential(
            torch.nn.Linear(channels_m, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, 1),
        )
        self.phi_h = torch.nn.Sequential(
            torch.nn.Linear(channels_h + channels_m, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, channels_h),
        )

    def forward(self, x, h, edge_attr, edge_index, c=None):
        """
        TODO
        """
        if c is None:
            c = degree(edge_index[0], x.shape[0]).unsqueeze(-1)
        return self.propagate(
            edge_index=edge_index, x=x, h=h, edge_attr=edge_attr, c=c
        )

    def message(self, x_i, x_j, h_i, h_j, edge_attr):
        """
        TODO
        """
        mh_ij = self.phi_e(
            torch.cat(
                [
                    h_i,
                    h_j,
                    torch.norm(x_i - x_j, dim=-1, keepdim=True) ** 2,
                    edge_attr,
                ],
                dim=-1,
            )
        )
        mx_ij = (x_i - x_j) * self.phi_x(mh_ij)
        return torch.cat((mx_ij, mh_ij), dim=-1)

    def update(self, aggr_out, x, h, edge_attr, c):
        """
        TODO
        """
        m_x, m_h = aggr_out[:, : self.m_len], aggr_out[:, self.m_len :]
        h_l1 = self.phi_h(torch.cat([h, m_h], dim=-1))
        x_l1 = x + (m_x / c)
        return x_l1, h_l1

    @property
    def edge_function(self):
        """
        TODO
        """
        return self._edge_function

    @property
    def attribute_function(self):
        """
        TODO
        """
        return self._attribute_function
