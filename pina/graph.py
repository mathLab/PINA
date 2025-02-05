from logging import warning

import torch

from . import LabelTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import inspect


class Graph:
    """
    Class for the graph construction.
    """

    def __init__(
            self,
            x,
            pos,
            edge_index,
            edge_attr=None,
            build_edge_attr=False,
            undirected=False,
            custom_build_edge_attr=None,
            additional_params=None
    ):
        """
        Constructor for the Graph class. This object creates a list of PyTorch Geometric Data objects.
        Based on the input of x and pos there could be the following cases:
            1. 1 pos, 1 x: a single graph will be created
            2. N pos, 1 x: N graphs will be created with the same node features
            3. 1 pos, N x: N graphs will be created with the same nodes but different node features
            4. N pos, N x: N graphs will be created

        :param x: Node features. Can be a single 2D tensor of shape [num_nodes, num_node_features],
                  or a 3D tensor of shape [n_graphs, num_nodes, num_node_features]
                  or a list of such 2D tensors of shape [num_nodes, num_node_features].
        :type x: torch.Tensor or list[torch.Tensor]
        :param pos: Node coordinates. Can be a single 2D tensor of shape [num_nodes, num_coordinates],
                    or a 3D tensor of shape [n_graphs, num_nodes, num_coordinates]
                    or a list of such 2D tensors of shape [num_nodes, num_coordinates].
        :type pos: torch.Tensor or list[torch.Tensor]
        :param edge_index: The edge index defining connections between nodes.
                           It should be a 2D tensor of shape [2, num_edges]
                           or a 3D tensor of shape [n_graphs, 2, num_edges]
                           or a list of such 2D tensors of shape [2, num_edges].
        :type edge_index: torch.Tensor or list[torch.Tensor]
        :param edge_attr: Edge features. If provided, should have the shape [num_edges, num_edge_features]
                          or be a list of such tensors for multiple graphs.
        :type edge_attr: torch.Tensor or list[torch.Tensor], optional
        :param build_edge_attr: Whether to compute edge attributes during initialization.
        :type build_edge_attr: bool, default=False
        :param undirected: If True, converts the graph(s) into an undirected graph by adding reciprocal edges.
        :type undirected: bool, default=False
        :param custom_build_edge_attr: A user-defined function to generate edge attributes dynamically.
                                       The function should take (x, pos, edge_index) as input and return a tensor
                                       of shape [num_edges, num_edge_features].
        :type custom_build_edge_attr: function or callable, optional
        :param additional_params: Dictionary containing extra attributes to be added to each Data object.
                                  Keys represent attribute names, and values should be tensors or lists of tensors.
        :type additional_params: dict, optional

        Note: if x, pos, and edge_index are both lists or 3D tensors, then len(x) == len(pos) == len(edge_index).
        """

        self.data = []
        x, pos, edge_index = self._check_input_consistency(x, pos, edge_index)

        # Check input dimension consistency and store the number of graphs
        data_len = self._check_len_consistency(x, pos)
        if inspect.isfunction(custom_build_edge_attr):
            self._build_edge_attr = custom_build_edge_attr

        # Check consistency and initialize additional_parameters (if present)
        additional_params = self._check_additional_params(additional_params,
                                                          data_len)

        # Make the graphs undirected
        if undirected:
            if isinstance(edge_index, list):
                edge_index = [to_undirected(e) for e in edge_index]
            else:
                edge_index = to_undirected(edge_index)

        # Prepare internal lists to create a graph list (same positions but
        # different node features)
        if isinstance(x, list) and isinstance(pos,
                                              (torch.Tensor, LabelTensor)):
            # Replicate the positions, edge_index and edge_attr
            pos, edge_index = [pos] * data_len, [edge_index] * data_len
        # Prepare internal lists to create a list containing a single graph
        elif isinstance(x, (torch.Tensor, LabelTensor)) and isinstance(pos, (
                torch.Tensor, LabelTensor)):
            # Encapsulate the input tensors into lists
            x, pos, edge_index = [x], [pos], [edge_index]
        # Prepare internal lists to create a list of graphs (same node features
        # but different positions)
        elif (isinstance(x, (torch.Tensor, LabelTensor))
              and isinstance(pos, list)):
            # Replicate the node features
            x = [x] * data_len
        elif not isinstance(x, list) and not isinstance(pos, list):
            raise TypeError("x and pos must be lists or tensors.")

        # Build the edge attributes
        edge_attr = self._check_and_build_edge_attr(edge_attr, build_edge_attr,
                                                    data_len, edge_index, pos,
                                                    x)

        # Perform the graph construction
        self._build_graph_list(x, pos, edge_index, edge_attr, additional_params)

    def _build_graph_list(self, x, pos, edge_index, edge_attr,
                          additional_params):
        for i, (x_, pos_, edge_index_) in enumerate(zip(x, pos, edge_index)):
            if isinstance(x_, LabelTensor):
                x_ = x_.tensor
            add_params_local = {k: v[i] for k, v in additional_params.items()}
            if edge_attr is not None:

                self.data.append(Data(x=x_, pos=pos_, edge_index=edge_index_,
                                      edge_attr=edge_attr[i],
                                      **add_params_local))
            else:
                self.data.append(Data(x=x_, pos=pos_, edge_index=edge_index_,
                                      **add_params_local))

    @staticmethod
    def _build_edge_attr(x, pos, edge_index):
        distance = torch.abs(pos[edge_index[0]] - pos[edge_index[1]])
        return distance

    @staticmethod
    def _check_len_consistency(x, pos):
        if isinstance(x, list) and isinstance(pos, list):
            if len(x) != len(pos):
                raise ValueError("x and pos must have the same length.")
            return max(len(x), len(pos))
        elif isinstance(x, list) and not isinstance(pos, list):
            return len(x)
        elif not isinstance(x, list) and isinstance(pos, list):
            return len(pos)
        else:
            return 1

    @staticmethod
    def _check_input_consistency(x, pos, edge_index=None):
        # If x is a 3D tensor, we split it into a list of 2D tensors
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            x = [x[i] for i in range(x.shape[0])]
        elif (not (isinstance(x, list) and all(t.ndim == 2 for t in x)) and
              not (isinstance(x, torch.Tensor) and x.ndim == 2)):
            raise TypeError("x must be either a list of 2D tensors or a 2D "
                            "tensor or a 3D tensor")

        # If pos is a 3D tensor, we split it into a list of 2D tensors
        if isinstance(pos, torch.Tensor) and pos.ndim == 3:
            pos = [pos[i] for i in range(pos.shape[0])]
        elif not (isinstance(pos, list) and all(
                t.ndim == 2 for t in pos)) and not (
                isinstance(pos, torch.Tensor) and pos.ndim == 2):
            raise TypeError("pos must be either a list of 2D tensors or a 2D "
                            "tensor or a 3D tensor")

        # If edge_index is a 3D tensor, we split it into a list of 2D tensors
        if edge_index is not None:
            if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 3:
                edge_index = [edge_index[i] for i in range(edge_index.shape[0])]
            elif not (isinstance(edge_index, list) and all(
                    t.ndim == 2 for t in edge_index)) and not (
                    isinstance(edge_index,
                               torch.Tensor) and edge_index.ndim == 2):
                raise TypeError(
                    "edge_index must be either a list of 2D tensors or a 2D "
                    "tensor or a 3D tensor")

        return x, pos, edge_index

    @staticmethod
    def _check_additional_params(additional_params, data_len):
        if additional_params is not None:
            if not isinstance(additional_params, dict):
                raise TypeError("additional_params must be a dictionary.")
            for param, val in additional_params.items():
                # Check if the values are tensors or lists of tensors
                if isinstance(val, torch.Tensor):
                    # If the tensor is 3D, we split it into a list of 2D tensors
                    # In this case there must be a additional parameter for each
                    # node
                    if val.ndim == 3:
                        additional_params[param] = [val[i] for i in
                                                    range(val.shape[0])]
                    # If the tensor is 2D, we replicate it for each node
                    elif val.ndim == 2:
                        additional_params[param] = [val] * data_len
                    # If the tensor is 1D, each graph has a scalar values as
                    # additional parameter
                    if val.ndim == 1:
                        if len(val) == data_len:
                            additional_params[param] = [val[i] for i in
                                                        range(len(val))]
                        else:
                            additional_params[param] = [val for _ in
                                                        range(data_len)]
                elif not isinstance(val, list):
                    raise TypeError("additional_params values must be tensors "
                                    "or lists of tensors.")
        else:
            additional_params = {}
        return additional_params

    def _check_and_build_edge_attr(self, edge_attr, build_edge_attr, data_len,
                                   edge_index, pos, x):
        # Check if edge_attr is consistent with x and pos
        if edge_attr is not None:
            if build_edge_attr is True:
                warning("edge_attr is not None. build_edge_attr will not be "
                        "considered.")
            if isinstance(edge_attr, list):
                if len(edge_attr) != data_len:
                    raise TypeError("edge_attr must have the same length as x "
                                     "and pos.")
            return [edge_attr] * data_len

        if build_edge_attr:
            return [self._build_edge_attr(x, pos_, edge_index_) for
                    pos_, edge_index_ in zip(pos, edge_index)]


class RadiusGraph(Graph):
    def __init__(
            self,
            x,
            pos,
            r,
            **kwargs
    ):
        x, pos, edge_index = Graph._check_input_consistency(x, pos)

        if isinstance(pos, (torch.Tensor, LabelTensor)):
            edge_index = RadiusGraph._radius_graph(pos, r)
        else:
            edge_index = [RadiusGraph._radius_graph(p, r) for p in pos]

        super().__init__(x=x, pos=pos, edge_index=edge_index,
                         **kwargs)

    @staticmethod
    def _radius_graph(points, r):
        """
        Implementation of the radius graph construction.
        :param points: The input points.
        :type points: torch.Tensor
        :param r: The radius.
        :type r: float
        :return: The edge index.
        :rtype: torch.Tensor
        """
        dist = torch.cdist(points, points, p=2)
        edge_index = torch.nonzero(dist <= r, as_tuple=False).t()
        return edge_index


class KNNGraph(Graph):
    def __init__(
            self,
            x,
            pos,
            k,
            **kwargs
    ):
        x, pos, edge_index = Graph._check_input_consistency(x, pos)
        if isinstance(pos, (torch.Tensor, LabelTensor)):
            edge_index = KNNGraph._knn_graph(pos, k)
        else:
            edge_index = [KNNGraph._knn_graph(p, k) for p in pos]
        super().__init__(x=x, pos=pos, edge_index=edge_index,
                         **kwargs)

    @staticmethod
    def _knn_graph(points, k):
        """
        Implementation of the k-nearest neighbors graph construction.
        :param points: The input points.
        :type points: torch.Tensor
        :param k: The number of nearest neighbors.
        :type k: int
        :return: The edge index.
        :rtype: torch.Tensor
        """
        dist = torch.cdist(points, points, p=2)
        knn_indices = torch.topk(dist, k=k + 1, largest=False).indices[:, 1:]
        row = torch.arange(points.size(0)).repeat_interleave(k)
        col = knn_indices.flatten()
        edge_index = torch.stack([row, col], dim=0)
        return edge_index
