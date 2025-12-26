import torch
from copy import deepcopy
from .condition_interface import ConditionInterface
from ..graph import Graph, LabelBatch
from ..label_tensor import LabelTensor
from ..data.dummy_dataloader import DummyDataloader
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from functools import partial


class ConditionBase(ConditionInterface):
    collate_fn_dict = {
        "tensor": torch.stack,
        "label_tensor": LabelTensor.stack,
        "graph": LabelBatch.from_data_list,
        "data": Batch.from_data_list,
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.data = self._store_data(**kwargs)

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
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

    def _store_tensor_data(self, **kwargs):
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

    def _store_graph_data(self, graphs, tensors=None, key=None):
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
        for i, graph in enumerate(graphs):
            new_graph = deepcopy(graph)
            tensor = tensors[i]
            setattr(new_graph, key, tensor)
            data.append(new_graph)
        return {"data": data}

    def _store_data(self, **kwargs):
        return self._store_tensor_data(**kwargs)

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}

    @classmethod
    def automatic_batching_collate_fn(cls, batch):
        """
        Collate function to be used in DataLoader.

        :param batch: A list of items from the dataset.
        :type batch: list
        :return: A collated batch.
        :rtype: dict
        """

        to_return = {key: [] for key in batch[0].keys()}
        for item in batch:
            for key, value in item.items():
                to_return[key].append(value)
        for key, values in to_return.items():
            collate_function = cls.collate_fn_dict.get(
                "label_tensor"
                if isinstance(values[0], LabelTensor)
                else (
                    "label_tensor"
                    if isinstance(values[0], torch.Tensor)
                    else "graph" if isinstance(values[0], Graph) else "data"
                )
            )
            to_return[key] = collate_function(values)
        return to_return

    @staticmethod
    def collate_fn(batch, condition):
        """
        Collate function for automatic batching to be used in DataLoader.

        :param batch: A list of items from the dataset.
        :type batch: list
        :return: A collated batch.
        :rtype: list
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
            return DummyDataloader(dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=(
                partial(self.collate_fn, condition=self)
                if not automatic_batching
                else self.automatic_batching_collate_fn
            ),
            # collate_fn = self.automatic_batching_collate_fn
        )
