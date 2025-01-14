import torch
import warnings
from . import LabelTensor
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class Graph:
    def __init__(self, x=None, pos=None, edge_index=None, edge_attr=None,
                 build_edge_attr=False, undirected=False, method=None,**kwargs):
        if edge_index is None:
            if method is None:
                raise ValueError("Cannot initialize graph object without edge index. "
                                 "Input edge_index or use set method and related params")
            edge_index = self._build_edge_index(pos, method, **kwargs)
        elif method is not None:
            warnings.warning("Both method and edge_index are not None. "
                             "Using edge_index provided in input")

        if undirected:
            if isinstance(edge_index, list):
                edge_index = [to_undirected(e) for e in edge_index]
            else:
                edge_index = to_undirected(edge_index)

        if build_edge_attr and edge_attr is None:
            edge_attr = self._build_edge_attr(pos, edge_index)

        if isinstance(x, list) and isinstance(pos,
                                                (torch.Tensor, LabelTensor)):
            pos, edge_index = [pos] * len(x), [edge_index] * len(x)
            if edge_attr is not None:
                edge_attr = [edge_attr] * len(x)
        elif isinstance(x, (torch.Tensor, LabelTensor)) and isinstance(pos, (
                torch.Tensor, LabelTensor)):
            x, pos, edge_index = [x], [pos], [edge_index]
            if edge_attr is not None:
                edge_attr = [edge_attr]
        elif (isinstance(x, (torch.Tensor, LabelTensor))
              and isinstance(pos, list)):
            x = [x] * len(pos)
        elif not isinstance(x, list) and not isinstance(pos, list):
            raise ValueError("x and pos must be lists or tensors.")

        self.data = []
        self.build_graph_list(x, pos, edge_index, edge_attr)

    def build_graph_list(self, x, pos, edge_index, edge_attr):
        for i, (x_, pos_, edge_index_) in enumerate(zip(x, pos, edge_index)):
            if isinstance(x_, LabelTensor):
                x_ = x_.tensor
            if edge_attr is not None:
                self.data.append(Data(x=x_, pos=pos_, edge_index=edge_index_,
                                      edge_attr=edge_attr[i]))
            else:
                self.data.append(Data(x=x_, pos=pos_, edge_index=edge_index_, ))

    @staticmethod
    def _build_edge_index(pos, method, **kwargs):
        if method == 'radius':
            func = radius_graph
        elif method == 'knn':
            func = knn_graph
        else:
            raise ValueError("The method must be 'radius' or 'knn.")

        if isinstance(pos, (torch.Tensor, LabelTensor)):
            if isinstance(pos, LabelTensor):
                pos = pos.tensor
            return func(pos, **kwargs)

        return [func(p, **kwargs) for p in pos]


    @staticmethod
    def _build_edge_attr(pos, edge_index):
        distance = torch.norm((pos[edge_index[0]] - pos[edge_index[1]]), dim=-1)
        return torch.cat([distance.unsqueeze(-1), pos[edge_index[0]], pos[edge_index[1]]], dim=-1)
