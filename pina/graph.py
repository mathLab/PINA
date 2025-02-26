"""
This module provides an interface to build torch_geometric.data.Data objects.
"""

import warnings

import torch

from . import LabelTensor
from .utils import check_consistency
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import inspect


class Graph:
    """
    A class to build torch_geometric.data.Data objects.
    """

    def __new__(
        cls,
        pos,
        edge_index,
        edge_attr=None,
        build_edge_attr=False,
        undirected=False,
        custom_build_edge_attr=None,
        **kwargs,
    ):
        """
        Initializes the Graph object. If x is provided, a Data object is
        returned; otherwise, a Graph object is returned. The Graph object
        allows constructing multiple graphs with the same structure.
        :param torch.Tensor pos: The position tensor.
        :param torch.Tensor edge_index: The edge index tensor.
        :param torch.Tensor edge_attr: The edge attribute tensor.
        :param bool build_edge_attr: Whether to build the edge attributes.
        :param bool undirected: Whether the graph is undirected.
        :param callable custom_build_edge_attr: A custom function to build the
        edge attributes.
        :param kwargs: Additional parameters.
        :return: The Data or Graph object.
        :rtype: torch_geometric.data.Data | Graph
        """

        instance = super().__new__(cls)
        instance._check_pos_consistency(pos)
        instance._check_edge_index_consistency(edge_index)
        if custom_build_edge_attr is not None:
            if not inspect.isfunction(custom_build_edge_attr):
                raise TypeError(
                    "custom_build_edge_attr must be a function or callable."
                )
            instance._build_edge_attr = custom_build_edge_attr

        if "x" not in kwargs:
            return instance

        x = kwargs.pop("x")
        instance._check_x_consistency(x, pos)
        edge_index = instance._preprocess_edge_index(edge_index, undirected)
        if build_edge_attr:
            if edge_attr is not None:
                warnings.warn(
                    "build_edge_attr is set to True, but edge_attr is not "
                    "None. The edge attributes will be computed."
                )
            edge_attr = instance._build_edge_attr(x, pos, edge_index)

        return Data(
            x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, **kwargs
        )

    def __init__(
        self, pos, edge_index, edge_attr=None, build_edge_attr=False, **kwargs
    ):
        """
        Initialize the Graph object.
        :param torch.Tensor pos: The position tensor.
        :param torch.Tensor edge_index: The edge index tensor.
        :param torch.Tensor edge_attr: The edge attribute tensor.
        :param bool build_edge_attr: Whether to build the edge attributes.
        :param kwargs: Additional parameters.
        """

        self.pos = pos
        self.edge_index = edge_index
        self.build_edge_attr = True
        if build_edge_attr:
            if edge_attr is not None:
                warnings.warn(
                    "build_edge_attr is set to True, but edge_attr is not "
                    "None. The edge attributes will be computed."
                )
        elif edge_attr is not None:
            self.edge_attr = edge_attr

        # Store additional parameters
        self.kwargs = kwargs

    def __call__(self, x, **kwargs):
        """
        Build a new Data object with the input tensor x.
        :param torch.Tensor x: The input tensor.
        :return: The new Data object.
        :rtype: torch_geometric.data.Data
        """
        self._check_x_consistency(x, self.pos)
        if not hasattr(self, "edge_attr"):
            edge_attr = self._build_edge_attr(x, self.pos, self.edge_index)
        else:
            edge_attr = self.edge_attr

        # Combine global additional parameters to the ones provided for the
        # single Data instance
        kwargs.update(self.kwargs)

        return Data(
            x,
            pos=self.pos,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            **kwargs,
        )

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
    def _check_x_consistency(x, pos):
        """
        Check if the input tensor x is consistent with the position tensor pos.
        :param torch.Tensor x: The input tensor.
        :param torch.Tensor pos: The position tensor.
        """
        if x is not None:
            check_consistency(x, (torch.Tensor, LabelTensor))
            if x.ndim != 2:
                raise ValueError("x must be a 2D tensor.")
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

    @staticmethod
    def _build_edge_attr(x, pos, edge_index):
        """
        Compute the edge attributes.
        :param torch.Tensor x: The input tensor.
        :param torch.Tensor pos: The position tensor.
        :param torch.Tensor edge_index: The edge index.
        :return: The edge attributes.
        :rtype: torch.Tensor
        """
        distance = torch.abs(
            pos[edge_index[0]] - pos[edge_index[1]]
        ).as_subclass(torch.Tensor)
        return distance


class RadiusGraph:
    """
    A class to build a radius graph in the form of torch_geometric.data.Data
    objects.
    """

    def __new__(cls, pos, r, **kwargs):
        """
        Initialize the RadiusGraph object.
        :param torch.Tensor pos: The position tensor.
        :param float r: The radius.
        :param kwargs: Additional parameters.
        :return: The Data object or an instance Graph class useful to create
        multiple torch_geometric.data.Data objects.
        :rtype: torch_geometric.data.Data | Graph
        """
        check_consistency(r, float)
        Graph._check_pos_consistency(pos)
        edge_index = cls._radius_graph(pos, r)
        return Graph(pos=pos, edge_index=edge_index, **kwargs)

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


class KNNGraph:
    """
    A class to build a k-nearest neighbors graph in the form of
    torch_geometric.data.Data objects.
    """

    def __new__(cls, pos, k, **kwargs):
        """
        Initialize the KNN graph object.
        :param torch.Tensor pos: The position tensor.
        :param float r: The radius.
        :param kwargs: Additional parameters.
        :return: The Data object or an instance Graph class useful to create
        multiple torch_geometric.data.Data objects.
        :rtype: torch_geometric.data.Data | Graph
        """
        check_consistency(k, int)
        Graph._check_pos_consistency(pos)
        edge_index = KNNGraph._knn_graph(pos, k)
        return Graph(pos=pos, edge_index=edge_index, **kwargs)

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
