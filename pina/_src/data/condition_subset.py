"""Utilities for handling condition dataset subsets."""

from torch_geometric.data import Batch
from pina._src.core.graph import LabelBatch, Graph


class _ConditionSubset:
    """
    Wrapper around a condition dataset restricted to a subset of indices.

    The class behaves similarly to :class:`torch.utils.data.Subset` and supports
    cyclic indexing together with optional automatic batching.

    :Example:

        >>> import torch
        >>> from pina.condition import Condition
        >>> from pina import LabelTensor
        >>> pts = LabelTensor(torch.randn(20, 2), labels=["x", "y"])
        >>> condition = Condition(input=pts)
        >>> subset = _ConditionSubset(condition, [0, 1, 2, 3, 4], True)
        >>> len(subset)
        5
    """

    def __init__(self, condition, indices, automatic_batching):
        """
        Initialization of the :class:`_ConditionSubset` class.

        :param BaseCondition condition: The underlying condition.
        :param list[int] indices: The list of indices identifying the subset
            samples.
        :param bool automatic_batching: Whether dataset items should be returned
            directly or as raw indices.
        """
        super().__init__()

        # Initialize the class attributes
        self.condition = condition
        self.indices = indices
        self.automatic_batching = automatic_batching

        # Actual number of samples contained in the subset
        self.dataset_length = len(self.indices)

        # Effective iterable length used and modified during batching
        self.iterable_length = self.dataset_length

    def __len__(self):
        """
        Return the effective iterable length of the subset.

        :return: The number of accessible elements in the subset.
        :rtype: int
        """
        return self.iterable_length

    def __getitem__(self, idx):
        """
        Retrieve an element from the subset.

        If the requested index exceeds the actual dataset size, cyclic indexing
        is applied through modulo wrapping. When automatic batching is disabled,
        the raw dataset index is returned instead of the corresponding sample.

        :param int idx: The position of the element inside the subset.
        :return: The dataset sample or raw dataset index depending on the
            batching configuration.
        :rtype: dict | int
        """
        # Apply cyclic indexing if the requested index exceeds the subset length
        if idx >= self.dataset_length:
            idx = idx % self.dataset_length

        # Fetch the corresponding dataset index from the list of indices
        idx = self.indices[idx]

        # Return the raw dataset index if automatic batching is disabled
        if not self.automatic_batching:
            return idx

        return self.condition[idx]

    def get_all_data(self):
        """
        Retrieve and aggregate all subset samples.

        If the returned data contains a ``"data"`` field composed of graph
        objects, the samples are merged into a single batched graph structure
        using the appropriate batching implementation.

        :return: The aggregated subset data.
        :rtype: dict
        """
        # Fetch the data corresponding to the subset indices
        data = self.condition[self.indices]

        # Data as a list of graph objects merged into a single batched graph
        if "data" in data and isinstance(data["data"], list):

            # Define the batching function
            batch_fn = (
                LabelBatch.from_data_list
                if isinstance(data["data"][0], Graph)
                else Batch.from_data_list
            )

            # Merge the list of graph objects into a single batched graph
            data["data"] = batch_fn(data["data"])
            data = {"input": data["data"], "target": data["data"].y}

        return data
