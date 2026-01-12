"""
Base class for conditions.
"""

from copy import deepcopy
from functools import partial
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from .condition_interface import ConditionInterface
from ..graph import Graph, LabelBatch
from ..label_tensor import LabelTensor


class TensorCondition:
    """
    Base class for tensor conditions.
    """

    def store_data(self, **kwargs):
        """
        Store data for standard tensor condition

        :param kwargs: Keyword arguments representing the data to be stored.
        :return: A dictionary containing the stored data.
        :rtype: dict
        """
        data = {}
        for key, value in kwargs.items():
            data[key] = value
        return data


class GraphCondition:
    """
    Base class for graph conditions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        example = kwargs.get(self.graph_field)[0]
        self.batch_fn = (
            LabelBatch.from_data_list
            if isinstance(example, Graph)
            else Batch.from_data_list
        )

    def store_data(self, **kwargs):
        """
        Store data for graph condition

        :param graphs: List of graphs to store data in.
        :type graphs: list[Graph] | list[Data]
        :param tensors: List of tensors to store in the graphs.
        :type tensors: list[torch.Tensor] | list[LabelTensor]
        :param key: Key under which to store the tensors in the graphs.
        :type key: str
        :return: A dictionary containing the stored data.
        :rtype: dict
        """
        data = []
        graphs = kwargs.get(self.graph_field)
        for i, graph in enumerate(graphs):
            new_graph = deepcopy(graph)
            for key in self.tensor_fields:
                tensor = kwargs[key][i]
                mapping_key = self.keys_map.get(key)
                setattr(new_graph, mapping_key, tensor)
            data.append(new_graph)
        return {"data": data}

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.get_multiple_data(idx)
        return {"data": self.data["data"][idx]}

    def get_multiple_data(self, indices):
        """
        Get multiple data items based on the provided indices.

        :param List[int] indices: List of indices to retrieve.
        :return: Dictionary containing 'input' and 'target' data.
        :rtype: dict
        """
        to_return_dict = {}
        data = self.batch_fn([self.data["data"][i] for i in indices])
        to_return_dict[self.graph_field] = data
        for key in self.tensor_fields:
            mapping_key = self.keys_map.get(key)
            y = getattr(data, mapping_key)
            delattr(data, mapping_key)  # Avoid duplication of y on GPU memory
            to_return_dict[key] = y
        return to_return_dict

    @classmethod
    def automatic_batching_collate_fn(cls, batch):
        """
        Collate function to be used in DataLoader.

        :param batch: A list of items from the dataset.
        :type batch: list
        :return: A collated batch.
        :rtype: dict
        """
        collated_graphs = super().automatic_batching_collate_fn(batch)["data"]
        to_return_dict = {}
        for key in cls.tensor_fields:
            mapping_key = cls.keys_map.get(key)
            tensor = getattr(collated_graphs, mapping_key)
            to_return_dict[key] = tensor
            delattr(collated_graphs, mapping_key)
        to_return_dict[cls.graph_field] = collated_graphs
        return to_return_dict


class ConditionBase(ConditionInterface):
    """
    Base abstract class for all conditions in PINA.
    This class provides common functionality for handling data storage,
    batching, and interaction with the associated problem.
    """

    collate_fn_dict = {
        "tensor": torch.stack,
        "label_tensor": LabelTensor.stack,
        "graph": LabelBatch.from_data_list,
        "data": Batch.from_data_list,
    }

    def __init__(self, **kwargs):
        """
        Initialization of the :class:`ConditionBase` class.

        :param kwargs: Keyword arguments representing the data to be stored.
        """
        super().__init__()
        self.data = self.store_data(**kwargs)

    @property
    def problem(self):
        """
        Return the problem associated with this condition.

        :return: Problem associated with this condition.
        :rtype: ~pina.problem.abstract_problem.AbstractProblem
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param pina.problem.abstract_problem.AbstractProblem value: The problem
            to associate with this condition
        """
        self._problem = value

    @staticmethod
    def _check_graph_list_consistency(data_list):
        """
        Check the consistency of the list of Data | Graph objects.
        The following checks are performed:

        - All elements in the list must be of the same type (either
          :class:`~torch_geometric.data.Data` or :class:`~pina.graph.Graph`).

        - All elements in the list must have the same keys.

        - The data type of each tensor must be consistent across all elements.

        - If a tensor is a :class:`~pina.label_tensor.LabelTensor`, its labels
          must also be consistent across all elements.

        :param data_list: The list of Data | Graph objects to check.
        :type data_list: list[Data] | list[Graph] | tuple[Data] | tuple[Graph]
        :raises ValueError: If the input types are invalid.
        :raises ValueError: If all elements in the list do not have the same
            keys.
        :raises ValueError: If the type of each tensor is not consistent across
            all elements in the list.
        :raises ValueError: If the labels of the LabelTensors are not consistent
            across all elements in the list.
        """
        # If the data is a Graph or Data object, perform no checks
        if isinstance(data_list, (Graph, Data)):
            return

        # Check all elements in the list are of the same type
        if not all(isinstance(i, (Graph, Data)) for i in data_list):
            raise ValueError(
                "Invalid input. Please, provide either Data or Graph objects."
            )

        # Store the keys, data types and labels of the first element
        data = data_list[0]
        keys = sorted(list(data.keys()))
        data_types = {name: tensor.__class__ for name, tensor in data.items()}
        labels = {
            name: tensor.labels
            for name, tensor in data.items()
            if isinstance(tensor, LabelTensor)
        }

        # Iterate over the list of Data | Graph objects
        for data in data_list[1:]:

            # Check that all elements in the list have the same keys
            if sorted(list(data.keys())) != keys:
                raise ValueError(
                    "All elements in the list must have the same keys."
                )

            # Iterate over the tensors in the current element
            for name, tensor in data.items():
                # Check that the type of each tensor is consistent
                if tensor.__class__ is not data_types[name]:
                    raise ValueError(
                        f"Data {name} must be a {data_types[name]}, got "
                        f"{tensor.__class__}"
                    )

                # Check that the labels of each LabelTensor are consistent
                if isinstance(tensor, LabelTensor):
                    if tensor.labels != labels[name]:
                        raise ValueError(
                            "LabelTensor must have the same labels"
                        )

    def __len__(self):
        """
        Return the number of data points in the condition.

        :return: Number of data points.
        :rtype: int
        """
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        """
        Return the data point(s) at the specified index.

        :param idx: Index(es) of the data point(s) to retrieve.
        :type idx: int | list[int]
        :return: Data point(s) at the specified index.
        """
        return {name: data[idx] for name, data in self.data.items()}

    @classmethod
    def automatic_batching_collate_fn(cls, batch):
        """
        Collate function for automatic batching to be used in DataLoader.
        :param batch: A list of items from the dataset.
        :type batch: list
        :return: A collated batch.
        :rtype: dict
        """
        if not batch:
            return {}
        keys = batch[0].keys()
        columns = zip(*[item.values() for item in batch])

        to_return = {}

        # 2. Process each column
        for key, values in zip(keys, columns):
            # Determine type based on the first sample only
            first_val = values[0]

            if isinstance(first_val, (LabelTensor, torch.Tensor)):
                lookup_key = "label_tensor"
            elif isinstance(first_val, Graph):
                lookup_key = "graph"
            else:
                lookup_key = "data"

            # Execute the specific collate function
            to_return[key] = cls.collate_fn_dict[lookup_key](list(values))

        return to_return

    @staticmethod
    def collate_fn(batch, condition):
        """
        Collate function for custom batching to be used in DataLoader.

        :param batch: A list of items from the dataset.
        :type batch: list
        :param condition: The condition instance.
        :type condition: ConditionBase
        :return: A collated batch.
        :rtype: dict
        """
        data = condition[batch]
        return data

    def create_dataloader(
        self, dataset, batch_size, shuffle, automatic_batching
    ):
        """
        Create a DataLoader for the condition.

        :param int batch_size: The batch size for the DataLoader.
        :param bool shuffle: Whether to shuffle the data. Default is ``False``.
        :return: The DataLoader for the condition.
        :rtype: torch.utils.data.DataLoader
        """
        if batch_size == len(dataset):
            pass  # will be updated in the near future
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=(
                partial(self.collate_fn, condition=self)
                if not automatic_batching
                else self.automatic_batching_collate_fn
            ),
        )
