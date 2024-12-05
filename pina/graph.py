import torch
from . import LabelTensor
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.data import Data


class Graph:
    def __init__(self, x=None, pos=None, edge_index=None, edge_attr=None,
                 build_edge_attr=False, **kwargs):
        if edge_index is None:
            if isinstance(pos, (torch.Tensor, LabelTensor)):
                edge_index = self._build_edge_index(pos, **kwargs)
                if build_edge_attr:
                    edge_attr = self._build_edge_attr(pos, edge_index)
            else:
                edge_index = [self._build_edge_index(p, **kwargs) for p in pos]
                if build_edge_attr:
                    edge_attr = [self._build_edge_attr(p, ei) for p, ei in
                                 zip(pos, edge_index)]

        if isinstance(x, list) and isinstance(pos, list):
            pass
        elif isinstance(x, list) and isinstance(pos,
                                                (torch.Tensor, LabelTensor)):
            pos, edge_index = [pos] * len(x), [edge_index] * len(x)
            if edge_attr is not None:
                edge_attr = [edge_attr] * len(x)
        elif isinstance(x, (torch.Tensor, LabelTensor)) and isinstance(pos, (
                torch.Tensor, LabelTensor)):
            x, pos, edge_index = [x], [pos], [edge_index]
            if edge_attr is not None:
                edge_attr = [edge_attr]
        elif isinstance(x, (torch.Tensor, LabelTensor)) and isinstance(pos,
                                                                       list):
            x = [x] * len(pos)
        else:
            raise ValueError(
                "The input must be either x and pos or edge_index.")

        self.data = []
        self.build_graph_list(x, pos, edge_index, edge_attr)

    def build_graph_list(self, x, pos, edge_index, edge_attr):
        for i, (x_, pos_, edge_index_) in enumerate(zip(x, pos, edge_index)):
            if edge_attr is not None:
                self.data.append(Data(x=x_, pos=pos_, edge_index=edge_index_,
                                      edge_attr=edge_attr[i]))
            else:
                self.data.append(Data(x=x_, pos=pos_, edge_index=edge_index_, ))

    @staticmethod
    def _build_edge_index(pos, method, **kwargs):
        if method == 'radius':
            return radius_graph(pos, **kwargs)
        elif method == 'knn':
            return knn_graph(pos, **kwargs)
        else:
            raise ValueError("The method must be 'radius'.")

    @staticmethod
    def _build_edge_attr(pos, edge_index, ):
        return torch.norm((pos[edge_index[0]] - pos[edge_index[1]]), dim=-1)
