import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Tanh
from pina.model import FeedForward
from torch.nn import Parameter, Linear, Sequential

class GraphIntegralKernel(MessagePassing):
    def __init__(self,
                 width,
                 kernel_width,
                 n_layers=0,
                 inner_size=None,
                 layers=None
                 ):
        super(GraphIntegralKernel, self).__init__(aggr='add')
        self.dense = FeedForward(input_dimensions=kernel_width,
                                 output_dimensions=width**2,
                                 n_layers=n_layers,
                                 inner_size=inner_size,
                                 layers=layers)
        self.dense = Linear(kernel_width, width**2)
        self.width = width
        self.W = Parameter(torch.rand(width, width))

    def message(self, x_j, edge_attr):
        x = self.dense(edge_attr).view(-1, self.width, self.width)
        return torch.einsum('bij,bj->bi', x, x_j)
    def update(self, aggr_out, x):
        aggr_out = aggr_out + torch.mm(x, self.W)
        return aggr_out

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

class GNO(torch.nn.Module):
    def __init__(self,
                 lifting_operator,
                 projection_operator,
                 edge_features,
                 n_layers=1,
                 kernel_n_layers=0,
                 kernel_inner_size=None,
                 kernel_layers=None
                 ):
        super(GNO, self).__init__()
        self.lifting_operator = lifting_operator
        self.projection_operator = projection_operator
        self.tanh = Tanh()
        self.kernels = torch.nn.ModuleList(
            [GraphIntegralKernel(width=lifting_operator.out_features,
                                 kernel_width=edge_features,
                                 n_layers=kernel_n_layers,
                                 inner_size=kernel_inner_size,
                                 layers=kernel_layers
                                 )
             for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_attr):
        x = self.lifting_operator(x)
        for kernel in self.kernels:
            x = kernel(x, edge_index, edge_attr)
            x = self.tanh(x)
        x = self.projection_operator(x)
        return x
