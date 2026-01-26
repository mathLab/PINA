"""
Module for managing data in conditions.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from pina import LabelTensor
from ..graph import Graph, LabelBatch
from ..equation.equation_interface import EquationInterface
from .batch_manager import _BatchManager


class _DataManager:
    """
    Abstract base class for data managers.

    This class dynamically selects between :class:`_TensorDataManager` and
    :class:`_GraphDataManager` based on the types of the input data.
    """

    def __new__(cls, **kwargs):
        """
        Dynamically instantiate the appropriate subclass based on the types
        of the input data.
            - If all values in ``kwargs`` are instances of
              :class:`torch.Tensor`, :class:`LabelTensor` then
              :class:`_TensorDataManager` is instantiated.
            - Otherwise, :class:`_GraphDataManager` is instantiated.

        :param dict kwargs: The keyword arguments containing the data.
        :return: An instance of :class:`_TensorDataManager` or
            :class:`_GraphDataManager`.
        :rtype: _TensorDataManager | _GraphDataManager
        """
        # If not called directly, proceed with normal instantiation
        if cls is not _DataManager:
            return super().__new__(cls)

        # Does the data contain only tensors/LabelTensors/Equations?
        is_tensor_only = all(
            isinstance(v, (torch.Tensor, LabelTensor, EquationInterface))
            for v in kwargs.values()
        )
        # Choose the appropriate subclass, GraphDataManager or TensorDataManager
        subclass = _TensorDataManager if is_tensor_only else _GraphDataManager
        return super().__new__(subclass)

    def __init__(self, **kwargs):
        """
        Initialize the data manager with the provided keyword arguments.

        :param dict kwargs: The keyword arguments containing the data.
        """
        self.keys = list(kwargs.keys())


class _TensorDataManager(_DataManager):
    """
    Data manager for tensor data. Handles data stored as `torch.Tensor` or
    `LabelTensor`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        """
        Return the number of samples in the tensor data manager.

        :return: Number of samples.
        :rtype: int
        """
        return self.data[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        """
        Return a data item or a subset of data items by index.

        :param idx: Index or indices of the data items to retrieve.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: A new :class:`_TensorDataManager` instance containing the
            selected data items.
        :rtype: _TensorDataManager
        """
        # Mapping efficiente degli elementi
        new_data = {
            k: (self.data[k][idx] if k in self.keys else self.data[k])
            for k in self.keys
        }
        return _TensorDataManager(**new_data)

    @staticmethod
    def create_batch(items):
        """
        Create a batch from a list of :class:`_TensorDataManager` items.

        :param list items: List of :class:`_TensorDataManager` items to batch.
        :return: A new :class:`_BatchManager` instance containing the batched
        data.
        :rtype: _BatchManager
        """
        if not items:
            return None
        first = items[0]
        batch_data = _BatchManager()

        for k in first.keys:
            vals = [it.data[k] for it in items]
            sample = vals[0]

            if isinstance(sample, (torch.Tensor, LabelTensor)):
                batch_fn = (
                    LabelTensor.stack
                    if isinstance(sample, LabelTensor)
                    else torch.stack
                )
                batch_data[k] = batch_fn(vals, dim=0)
            else:
                batch_data[k] = sample
        return batch_data

    def to_batch(self):
        """
        Create a batch from the current tensor data manager.

        :return: A new :class:`_BatchManager` instance containing the batched
        data.
        :rtype: _BatchManager
        """
        batch_data = _BatchManager()
        for k in self.keys:
            batch_data[k] = self.data[k]
        return batch_data


class _GraphDataManager(_DataManager):
    """
    Data manager for graph data. Handles data stored as :class:`Graph`,
    :class:`Data`, or lists/tuples of these types. Moreover , it can also manage
    associated tensors stored as :class:`torch.Tensor` or :class:`LabelTensor`.
    """

    def __init__(self, **kwargs):
        """
        Initialize the graph data manager with the provided keyword arguments.

        :param dict kwargs: The keyword arguments containing the data.
        """
        super().__init__(**kwargs)
        self.graph_key = next(
            k
            for k, v in kwargs.items()
            if isinstance(v, (Graph, Data, list, tuple))
        )

        self.keys = [
            k
            for k in self.keys
            if k != self.graph_key
            and isinstance(kwargs[k], (torch.Tensor, LabelTensor))
        ]

        # Prepare graphs and assign tensors
        self.data = self._prepare_graphs(kwargs)

    def _prepare_graphs(self, kwargs):
        """
        Store tensors in the corresponding graphs.

        :param dict kwargs: The keyword arguments containing the graphs and
            associated tensors.
        :return: A list of graphs with tensors assigned.
        :rtype: list[Graph] | list[Data]
        """
        graphs = kwargs.pop(self.graph_key)
        if not isinstance(graphs, (list, tuple)):
            graphs = [graphs]

        n_graphs = len(graphs)
        for name, tensor in kwargs.items():
            # Verify consistency between number of graphs and tensor samples
            if n_graphs != tensor.shape[0]:
                raise ValueError(
                    f"Number of graphs ({n_graphs}) does not match "
                    f"number of samples for key '{name}' "
                    f"({kwargs[name].shape[0]})."
                )
            # Assign tensors to graphs
            for i, g in enumerate(graphs):
                setattr(g, name, tensor[i])

        return graphs

    def __len__(self):
        """
        Return the number of graphs in the graph data manager.

        :return: Number of graphs.
        :rtype: int
        """
        return len(self.data)

    def __getattr__(self, name):
        """
        Override attribute access to retrieve tensors or graphs. If the graph
        key is requested, return the list of graphs. If a tensor key is
        requested, stack the tensors from all graphs and return the result.

        :param str name: The name of the attribute to retrieve.
        :return: The requested tensor or graph.
        :rtype: torch.Tensor | LabelTensor | Graph | list[Graph] | Data |
        """
        # If the requested attribute is a tensor key, stack the tensors from
        # all graphs
        if name in self.keys:
            tensors = [getattr(g, name) for g in self.data]
            batch_fn = (
                LabelTensor.stack
                if isinstance(tensors[0], LabelTensor)
                else torch.stack
            )
            return batch_fn(tensors)

        # If the requested attribute is the graph key, return the graphs
        if name == self.graph_key:
            return self.data if len(self.data) > 1 else self.data[0]

        return super().__getattribute__(name)

    @classmethod
    def _init_from_graphs_list(cls, graphs, graph_key, keys):
        """
        Initialize a :class:`_GraphDataManager` instance from a list of graphs.
        This is used internally to create subsets of the data manager, without
        going through the full initialization process.

        :param list graphs: List of graphs to initialize the data manager with.
        :param str graph_key: Key under which the graphs are stored.
        :param list keys: List of tensor keys associated with the graphs.
        :return: A new :class:`_GraphDataManager` instance.
        :rtype: _GraphDataManager
        """
        # Create a new instance without calling __init__
        obj = _GraphDataManager.__new__(_GraphDataManager)
        obj.graph_key = graph_key
        obj.keys = keys
        obj.data = graphs
        return obj

    def __getitem__(self, idx):
        """
        Retrieve a graph or a subset of graphs by index.

        :param idx: Index or indices of the graphs to retrieve.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: A new :class:`_GraphDataManager` instance containing the
            selected graphs.
        :rtype: _GraphDataManager
        """
        # Manage int and slice directly
        if isinstance(idx, (int, slice)):
            selected = self.data[idx]
        # Manage list or tensor of indices
        elif isinstance(idx, (list, torch.Tensor)):
            selected = [self.data[i] for i in idx]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        # Ensure selected is a list
        if not isinstance(selected, list):
            selected = [selected]

        # Return a new _GraphDataManager instance with the selected graphs
        return _GraphDataManager._init_from_graphs_list(
            selected,
            # tensor_keys=self._tensor_keys,
            graph_key=self.graph_key,
            keys=self.keys,
        )

    def to_batch(self):
        """
        Create a batch from the current graph data manager.

        :return: A new :class:`_BatchManager` instance containing the batched
        data.
        :rtype: _BatchManager
        """
        batching_fn = (
            LabelBatch.from_data_list
            if isinstance(self.data[0], Graph)
            else Batch.from_data_list
        )

        batched_graph = batching_fn(self.data)
        batch_data = _BatchManager()
        for k in self.keys:
            if k == self.graph_key:
                continue
            batch_data[k] = getattr(batched_graph, k)
            delattr(batched_graph, k)
        batch_data[self.graph_key] = batched_graph
        return batch_data

    @staticmethod
    def create_batch(items):
        """
        Optimized batch creation.
        """
        if not items:
            return None

        first = items[0]
        graph_key = first.graph_key
        # Determine batching function once
        is_labeled = isinstance(first.data[0], Graph)
        batching_fn = (
            LabelBatch.from_data_list if is_labeled else Batch.from_data_list
        )

        # Efficient list comprehension for extraction
        # If to_batch() is called on self, self.data might be a list already.
        # If _create_batch is called on multiple managers, we grab the first
        # graph from each.
        graphs_to_batch = [item.data[0] for item in items]
        batched_graph = batching_fn(graphs_to_batch)

        batch_data = _BatchManager()

        # Use a set for O(1) lookups if keys is large
        keys_to_transfer = set(first.keys)
        if graph_key in keys_to_transfer:
            keys_to_transfer.remove(graph_key)

        for k in keys_to_transfer:
            # Check if attribute exists once to avoid AttributeError overhead
            val = getattr(batched_graph, k, None)
            if val is not None:
                batch_data[k] = val
                delattr(batched_graph, k)

        batch_data[graph_key] = batched_graph
        return batch_data
