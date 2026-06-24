"""Module for the Graph-Data Manager class."""

import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.graph import Graph, LabelBatch
from pina._src.data.manager.batch_manager import _BatchManager
from pina._src.data.manager.data_manager_interface import _DataManagerInterface


class _GraphDataManager(_DataManagerInterface):
    """
    Data manager for graph-based data. It handles inputs stored as
    :class:`Graph`, :class:`Data`, or lists / tuples of these types.

    :Example:

        >>> import torch
        >>> from pina.graph import Graph
        >>> graph = Graph(pos=torch.randn(5, 2),
        ...     edge_index=torch.tensor([[0, 1], [1, 0]]))
        >>> manager = _GraphDataManager(data=graph, target=torch.randn(5, 1))
        >>> len(manager)
        1
    """

    def __init__(self, **kwargs):
        """
        Initialization of the :class:`_GraphDataManager` class.

        :param dict kwargs: The keyword arguments for the graph data manager.
        """
        # Initialize keys
        self.keys = list(kwargs.keys())

        # Find graph-based data
        self.graph_key = next(
            k
            for k, v in kwargs.items()
            if isinstance(v, (Graph, Data, list, tuple))
        )

        # Find tensor data
        self.keys = [
            k
            for k in self.keys
            if k != self.graph_key
            and isinstance(kwargs[k], (torch.Tensor, LabelTensor))
        ]

        # Prepare graphs and assign tensors
        self.data = self._prepare_graphs(kwargs)

    def __len__(self):
        """
        Return the number of samples in the graph data manager.

        :return: The number of samples.
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the item at the specified indices.

        :param idx: The indices of the graphs to retrieve.
        :type idx: int | slice | list[int] | torch.Tensor
        :raises TypeError: If an index with invalid type is passed.
        :return: A new :class:`_GraphDataManager` instance containing the
            selected graphs.
        :rtype: _GraphDataManager
        """
        # Selection for integers or slices
        if isinstance(idx, (int, slice)):
            selected = self.data[idx]

        # Selection for lists or tensors
        elif isinstance(idx, (list, torch.Tensor)):
            selected = [self.data[i] for i in idx]

        # Raise TypeError if index type is invalid
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        # Ensure selected is a list
        if not isinstance(selected, list):
            selected = [selected]

        return _GraphDataManager._init_from_graphs_list(
            selected, graph_key=self.graph_key, keys=self.keys
        )

    def __getattr__(self, name):
        """
        Provide dynamic access to stored graph and tensor data.

        If ``name`` corresponds to the graph key, return the list of graph
        objects. If it matches a tensor key, retrieve the corresponding
        tensors from all graphs and stack them along the batch dimension.

        :param str name: The name of the attribute to access.
        :return: The requested graph data or stacked tensor values.
        :rtype: torch.Tensor | LabelTensor | list[Graph] | list[Data]
        """
        # Stack tensors from all graph if name is a tensor key
        if name in self.keys:
            tensors = [getattr(g, name) for g in self.data]
            batch_fn = (
                LabelTensor.stack
                if isinstance(tensors[0], LabelTensor)
                else torch.stack
            )
            return batch_fn(tensors)

        # Otherwise, return graphs
        if name == self.graph_key:
            return self.data if len(self.data) > 1 else self.data[0]

        return super().__getattribute__(name)

    def _prepare_graphs(self, kwargs):
        """
        Attach tensor data to the corresponding graph objects.

        :param kwargs: The keyword arguments containing graph data and
            associated tensor features.
        :raises ValueError: If the number of graphs does not match the number of
            samples in the tensor of features to associate.
        :return: A list of graphs with the corresponding tensors assigned.
        :rtype: list[Graph] | list[Data]
        """
        # Get graph-based data and store in a list
        graphs = kwargs.pop(self.graph_key)
        if not isinstance(graphs, (list, tuple)):
            graphs = [graphs]

        # Iterate of items
        for name, tensor in kwargs.items():

            # Verify the consistency between the number of graphs and samples
            if len(graphs) != tensor.shape[0]:
                raise ValueError(
                    f"Number of graphs ({len(graphs)}) does not match "
                    f"number of samples for key '{name}' "
                    f"({kwargs[name].shape[0]})."
                )

            # Assign tensors to graphs
            for i, g in enumerate(graphs):
                setattr(g, name, tensor[i])

        return graphs

    def to_batch(self):
        """
        Create a batch from the current graph data manager.

        :return: A new instance of :class:`_BatchManager` with batched data.
        :rtype: _BatchManager
        """
        # Define the batch function
        batching_fn = (
            LabelBatch.from_data_list
            if isinstance(self.data[0], Graph)
            else Batch.from_data_list
        )

        # Create the batch manager
        batch_data = _BatchManager()
        batched_graph = batching_fn(self.data)
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
        Create a batch from a list of :class:`_GraphDataManager` items.

        :param list[_GraphDataManager] items: A list of
            :class:`_GraphDataManager` items to batch.
        :return: A new instance of :class:`_BatchManager` containing the batched
            data.
        :rtype: _BatchManager
        """
        # Return None if no items are provided
        if not items:
            return None

        # Retrieve the first _GraphDataManager of the list and corresponding key
        first = items[0]
        graph_key = first.graph_key

        # Initialize the batch manager
        batch_data = _BatchManager()

        # Define batch function
        batching_fn = (
            LabelBatch.from_data_list
            if isinstance(first.data[0], Graph)
            else Batch.from_data_list
        )

        # Batch over graphs
        batched_graph = batching_fn([item.data[0] for item in items])

        # Use a set for O(1) lookups if keys are large
        keys_to_transfer = set(first.keys)
        if graph_key in keys_to_transfer:
            keys_to_transfer.remove(graph_key)

        # Iterate over the keys of the _GraphDataManager
        for k in keys_to_transfer:

            # Extract values
            val = getattr(batched_graph, k, None)
            if val is not None:
                batch_data[k] = val
                delattr(batched_graph, k)

        # Assign key to batch
        batch_data[graph_key] = batched_graph

        return batch_data

    @classmethod
    def _init_from_graphs_list(cls, graphs, graph_key, keys):
        """
        Create a :class:`_GraphDataManager` instance directly from a list of
        graph objects.

        This method bypasses the standard initialization logic and is used
        internally to construct new instances (e.g., subsets) from already
        processed graph data.

        :param list graphs: A list of graph objects.
        :param str graph_key: The name of the attribute used to store the
            graphs.
        :param list keys: A list of tensor keys associated with the graphs.
        :return: A new instance of :class:`_GraphDataManager`.
        :rtype: _GraphDataManager
        """
        # Create a new instance without calling __init__
        obj = _GraphDataManager.__new__(_GraphDataManager)
        obj.graph_key = graph_key
        obj.keys = keys
        obj.data = graphs

        return obj
