from logging import warning

import torch
from . import LabelTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class Graph:
    """
    Class for the graph construction.
    """

    def __init__(self,
                 x,
                 pos,
                 edge_index,
                 edge_attr=None,
                 build_edge_attr=False,
                 undirected=False,
                 additional_params=None):
        """
        Constructor for the Graph class.
        :param x: The node features.
        :type x: torch.Tensor or list[torch.Tensor]
        :param pos: The node positions.
        :type pos: torch.Tensor or list[torch.Tensor]
        :param edge_index: The edge index.
        :type edge_index: torch.Tensor or list[torch.Tensor]
        :param edge_attr: The edge attributes.
        :type edge_attr: torch.Tensor or list[torch.Tensor]
        :param build_edge_attr: Whether to build the edge attributes.
        :type build_edge_attr: bool
        :param undirected: Whether to build an undirected graph.
        :type undirected: bool
        :param additional_params: Additional parameters.
        :type additional_params: dict
        """
        self.data = []
        x, pos, edge_index = Graph._check_input_consistency(x, pos, edge_index)

        # Check input dimension consistency and store the number of graphs
        data_len = self._check_len_consistency(x, pos)

        # Initialize additional_parameters (if present)
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

        # Make the graphs undirected
        if undirected:
            if isinstance(edge_index, list):
                edge_index = [to_undirected(e) for e in edge_index]
            else:
                edge_index = to_undirected(edge_index)

        if build_edge_attr:
            if edge_attr is not None:
                warning("Edge attributes are provided, build_edge_attr is set "
                        "to True. The provided edge attributes will be ignored.")
            edge_attr = self._build_edge_attr(pos, edge_index)

        # Prepare internal lists to create a graph list (same positions but
        # different node features)
        if isinstance(x, list) and isinstance(pos,
                                              (torch.Tensor, LabelTensor)):
            # Replicate the positions, edge_index and edge_attr
            pos, edge_index = [pos] * data_len, [edge_index] * data_len
            if edge_attr is not None:
                edge_attr = [edge_attr] * data_len
        # Prepare internal lists to create a list containing a single graph
        elif isinstance(x, (torch.Tensor, LabelTensor)) and isinstance(pos, (
                torch.Tensor, LabelTensor)):
            # Encapsulate the input tensors into lists
            x, pos, edge_index = [x], [pos], [edge_index]
            if isinstance(edge_attr, torch.Tensor):
                edge_attr = [edge_attr]
        # Prepare internal lists to create a list of graphs (same node features
        # but different positions)
        elif (isinstance(x, (torch.Tensor, LabelTensor))
              and isinstance(pos, list)):
            # Replicate the node features
            x = [x] * data_len
        elif not isinstance(x, list) and not isinstance(pos, list):
            raise TypeError("x and pos must be lists or tensors.")

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
    def _build_edge_attr(pos, edge_index):
        if isinstance(pos, torch.Tensor):
            pos = [pos]
            edge_index = [edge_index]
        distance = [pos_[edge_index_[0]] - pos_[edge_index_[1]] ** 2 for
                    pos_, edge_index_ in zip(pos, edge_index)]
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

        # If pos is a 3D tensor, we split it into a list of 2D tensors
        if isinstance(pos, torch.Tensor) and pos.ndim == 3:
            pos = [pos[i] for i in range(pos.shape[0])]

        # If edge_index is a 3D tensor, we split it into a list of 2D tensors
        if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 3:
            edge_index = [edge_index[i] for i in range(edge_index.shape[0])]
        return x, pos, edge_index


class RadiusGraph(Graph):
    def __init__(self,
                 x,
                 pos,
                 r,
                 build_edge_attr=False,
                 undirected=False,
                 additional_params=None, ):
        x, pos, edge_index = Graph._check_input_consistency(x, pos)

        if isinstance(pos, (torch.Tensor, LabelTensor)):
            edge_index = RadiusGraph._radius_graph(pos, r)
        else:
            edge_index = [RadiusGraph._radius_graph(p, r) for p in pos]

        super().__init__(x=x, pos=pos, edge_index=edge_index,
                         build_edge_attr=build_edge_attr,
                         undirected=undirected,
                         additional_params=additional_params)

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
    def __init__(self,
                 x,
                 pos,
                 k,
                 build_edge_attr=False,
                 undirected=False,
                 additional_params=None,
                 ):
        x, pos, edge_index = Graph._check_input_consistency(x, pos)
        if isinstance(pos, (torch.Tensor, LabelTensor)):
            edge_index = KNNGraph._knn_graph(pos, k)
        else:
            edge_index = [KNNGraph._knn_graph(p, k) for p in pos]
        super().__init__(x=x, pos=pos, edge_index=edge_index,
                         build_edge_attr=build_edge_attr,
                         undirected=undirected,
                         additional_params=additional_params)

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
