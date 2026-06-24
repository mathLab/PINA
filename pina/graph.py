"""Public API for Graph connectivity and neighborhood logic.

This module exposes core graph types used to define spatial relationships
between points, such as fixed-radius and k-nearest neighbor (KNN) structures.

:Example:

    >>> from pina.graph import KNNGraph, RadiusGraph
    >>> import torch
    >>> graph = KNNGraph(k=3)
    >>> x = torch.rand(10, 2)
    >>> edge_index = graph(x)
    >>> edge_index.shape
    torch.Size([2, 30])
"""

from pina._src.core.graph import Graph, RadiusGraph, KNNGraph

__all__ = [
    "Graph",
    "RadiusGraph",
    "KNNGraph",
]
