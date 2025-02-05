import torch
from torch.nn import Tanh
from .layers import GNOBlock
from .base_no import KernelNeuralOperator


class GraphNeuralKernel(torch.nn.Module):
    """
    TODO add docstring
    """

    def __init__(
            self,
            width,
            edge_features,
            n_layers=2,
            internal_n_layers=0,
            internal_layers=None,
            inner_size=None,
            internal_func=None,
            external_func=None,
            shared_weights=False
    ):
        """
        The Graph Neural Kernel constructor.

        :param width: The width of the kernel.
        :type width: int
        :param edge_features: The number of edge features.
        :type edge_features: int
        :param n_layers: The number of kernel layers.
        :type n_layers: int
        :param internal_n_layers: The number of layers the FF Neural Network internal to each Kernel Layer.
        :type internal_n_layers: int
        :param internal_layers: Number of neurons of hidden layers(s) in the FF Neural Network inside for each Kernel Layer.
        :type internal_layers: list | tuple
        :param internal_func: The activation function used inside the computation of the representation of the edge features in the Graph Integral Layer.
        :param external_func: The activation function applied to the output of the Graph Integral Layer.
        :type external_func: torch.nn.Module
        :param shared_weights: If ``True`` the weights of the Graph Integral Layers are shared.
        """
        super().__init__()
        if external_func is None:
            external_func = Tanh
        if internal_func is None:
            internal_func = Tanh

        if shared_weights:
            self.layers = GNOBlock(
                width=width,
                edges_features=edge_features,
                n_layers=internal_n_layers,
                layers=internal_layers,
                inner_size=inner_size,
                internal_func=internal_func,
                external_func=external_func)
            self.n_layers = n_layers
            self.forward = self.forward_shared
        else:
            self.layers = torch.nn.ModuleList(
                [GNOBlock(
                    width=width,
                    edges_features=edge_features,
                    n_layers=internal_n_layers,
                    layers=internal_layers,
                    inner_size=inner_size,
                    internal_func=internal_func,
                    external_func=external_func
                )
                    for _ in range(n_layers)]
            )

    def forward(self, x, edge_index, edge_attr):
        """
        The forward pass of the Graph Neural Kernel used when the weights are not shared.

        :param x: The input batch.
        :type x: torch.Tensor
        :param edge_index: The edge index.
        :type edge_index: torch.Tensor
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x

    def forward_shared(self, x, edge_index, edge_attr):
        """
        The forward pass of the Graph Neural Kernel used when the weights are shared.

        :param x: The input batch.
        :type x: torch.Tensor
        :param edge_index: The edge index.
        :type edge_index: torch.Tensor
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor
        """
        for _ in range(self.n_layers):
            x = self.layers(x, edge_index, edge_attr)
        return x


class GraphNeuralOperator(KernelNeuralOperator):
    """
    TODO add docstring
    """

    def __init__(
            self,
            lifting_operator,
            projection_operator,
            edge_features,
            n_layers=10,
            internal_n_layers=0,
            inner_size=None,
            internal_layers=None,
            internal_func=None,
            external_func=None,
            shared_weights=True
    ):
        """
        The Graph Neural Operator constructor.

        :param lifting_operator: The lifting operator mapping the node features to its hidden dimension.
        :type lifting_operator: torch.nn.Module
        :param projection_operator: The projection operator mapping the hidden representation of the nodes features to the output function.
        :type projection_operator: torch.nn.Module
        :param edge_features: Number of edge features.
        :type edge_features: int
        :param n_layers: The number of kernel layers.
        :type n_layers: int
        :param internal_n_layers: The number of layers the Feed Forward Neural Network internal to each Kernel Layer.
        :type internal_n_layers: int
        :param internal_layers: Number of neurons of hidden layers(s) in the FF Neural Network inside for each Kernel Layer.
        :type internal_layers: list | tuple
        :param internal_func: The activation function used inside the computation of the representation of the edge features in the Graph Integral Layer.
        :type internal_func: torch.nn.Module
        :param external_func: The activation function applied to the output of the Graph Integral Kernel.
        :type external_func: torch.nn.Module
        :param shared_weights: If ``True`` the weights of the Graph Integral Layers are shared.
        :type shared_weights: bool
        """

        if internal_func is None:
            internal_func = Tanh
        if external_func is None:
            external_func = Tanh

        super().__init__(
            lifting_operator=lifting_operator,
            integral_kernels=GraphNeuralKernel(
                width=lifting_operator.out_features,
                edge_features=edge_features,
                internal_n_layers=internal_n_layers,
                inner_size=inner_size,
                internal_layers=internal_layers,
                external_func=external_func,
                internal_func=internal_func,
                n_layers=n_layers,
                shared_weights=shared_weights
            ),
            projection_operator=projection_operator
        )

    def forward(self, x):
        """
        The forward pass of the Graph Neural Operator.

        :param x: The input batch.
        :type x: torch_geometric.data.Batch
        """
        x, edge_index, edge_attr = x.x, x.edge_index, x.edge_attr
        x = self.lifting_operator(x)
        x = self.integral_kernels(x, edge_index, edge_attr)
        x = self.projection_operator(x)
        return x
