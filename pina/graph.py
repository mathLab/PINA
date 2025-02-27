"""
This module provides an interface to build torch_geometric.data.Data objects.
"""

import warnings

import torch

from . import LabelTensor
from .utils import check_consistency, is_function
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class Graph(Data):
    """
    A class to build torch_geometric.data.Data objects.
    """

    def __new__(
        cls,
        **kwargs,
    ):
        """
        :param kwargs: Parameters to construct the Graph object.
        :return: The Data object.
        :rtype: torch_geometric.data.Data
        """
        # create class instance
        instance = Data.__new__(cls)

        # check the consistency of types defined in __init__, the others are not
        # checked (as in pyg Data object)
        instance._check_type_consistency(**kwargs)
        
        return instance
    
    def __init__(
        self,
        x=None,
        edge_index=None,
        pos=None,
        edge_attr=None,
        undirected=False,
        **kwargs,
    ):
        """
        Initialize the Graph object.
        :param torch.Tensor pos: The position tensor.
        :param torch.Tensor edge_index: The edge index tensor.
        :param torch.Tensor edge_attr: The edge attribute tensor.
        :param bool build_edge_attr: Whether to build the edge attributes.
        :param kwargs: Additional parameters.
        """
        # preprocessing
        self._preprocess_edge_index(edge_index, undirected)

        # calling init
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr,
                         pos=pos, **kwargs)

    def _check_type_consistency(self, **kwargs):
        # default types, specified in cls.__new__, by default they are Nont
        # if specified in **kwargs they get override
        x, pos, edge_index, edge_attr = None, None, None, None
        if "pos" in kwargs:
            pos = kwargs["pos"]
            self._check_pos_consistency(pos)
        if "edge_index" in kwargs:
            edge_index = kwargs["edge_index"]
            self._check_edge_index_consistency(edge_index)
        if "x" in kwargs:
            x = kwargs["x"]
            self._check_x_consistency(x, pos)
        if "edge_attr" in kwargs:
            edge_attr = kwargs["edge_attr"]
            self._check_edge_attr_consistency(edge_attr, edge_index)
        if "undirected" in kwargs:
            undirected = kwargs["undirected"]
            check_consistency(undirected, bool)

    @staticmethod
    def _check_pos_consistency(pos):
        """
        Check if the position tensor is consistent.
        :param torch.Tensor pos: The position tensor.
        """
        check_consistency(pos, (torch.Tensor, LabelTensor))
        if pos.ndim != 2:
            raise ValueError("pos must be a 2D tensor.")

    @staticmethod
    def _check_edge_index_consistency(edge_index):
        """
        Check if the edge index is consistent.
        :param torch.Tensor edge_index: The edge index tensor.
        """
        check_consistency(edge_index, (torch.Tensor, LabelTensor))
        if edge_index.ndim != 2:
            raise ValueError("edge_index must be a 2D tensor.")
        if edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

    @staticmethod
    def _check_edge_attr_consistency(edge_attr, edge_index):
        """
        Check if the edge attr is consistent.
        :param torch.Tensor edge_attr: The edge attribute tensor.

        :param torch.Tensor edge_index: The edge index tensor.
        """
        check_consistency(edge_attr, (torch.Tensor, LabelTensor))
        if edge_attr.ndim != 2:
            raise ValueError("edge_attr must be a 2D tensor.")
        if edge_attr.size(1) != edge_index.size(0):
            raise ValueError(
                "edge_attr must have shape "
                "[num_edges, num_edge_features], expected "
                f"num_edges {edge_index.size(0)} "
                f"got {edge_attr.size(1)}."
            )

    @staticmethod
    def _check_x_consistency(x, pos=None):
        """
        Check if the input tensor x is consistent with the position tensor pos.
        :param torch.Tensor x: The input tensor.
        :param torch.Tensor pos: The position tensor.
        """
        if x is not None:
            check_consistency(x, (torch.Tensor, LabelTensor))
            if x.ndim != 2:
                raise ValueError("x must be a 2D tensor.")
            if pos is not None:
                if x.size(0) != pos.size(0):
                    raise ValueError("Inconsistent number of nodes.")

    @staticmethod
    def _preprocess_edge_index(edge_index, undirected):
        """
        Preprocess the edge index.
        :param torch.Tensor edge_index: The edge index.
        :param bool undirected: Whether the graph is undirected.
        :return: The preprocessed edge index.
        :rtype: torch.Tensor
        """
        if undirected:
            edge_index = to_undirected(edge_index)
        return edge_index

class RadiusGraph(Graph):
    def __init__(
        self,
        radius,
        x=None,
        pos=None,
        edge_attr=None,
        undirected=False,
        **kwargs,
    ):
        super().__init__(x=x, edge_index=None, edge_attr=edge_attr,
                         pos=pos, undirected=undirected, **kwargs)
        edge_index = self._radius_graph(pos, radius)
        self.radius = radius
        self.edge_index = edge_index
    
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
        if isinstance(edge_index, LabelTensor):
            edge_index = edge_index.tensor
        return edge_index
    
class KNNGraph(Graph):
    def __init__(
        self,
        neighboors,
        x=None,
        pos=None,
        edge_attr=None,
        undirected=False,
        **kwargs,
    ):
        super().__init__(x=x, edge_index=None, edge_attr=edge_attr,
                         pos=pos, undirected=undirected, **kwargs)
        edge_index = self._knn_graph(pos, neighboors)
        self.neighboors = neighboors
        self.edge_index = edge_index
    
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
        if isinstance(points, LabelTensor):
            points = points.tensor
        dist = torch.cdist(points, points, p=2)
        knn_indices = torch.topk(dist, k=k + 1, largest=False).indices[:, 1:]
        row = torch.arange(points.size(0)).repeat_interleave(k)
        col = knn_indices.flatten()
        edge_index = torch.stack([row, col], dim=0)
        if isinstance(edge_index, LabelTensor):
            edge_index = edge_index.tensor
        return edge_index