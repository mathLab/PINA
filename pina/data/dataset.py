"""Module for the PINA dataset classes."""

from torch.utils.data import Dataset
from torch_geometric.data import Data
from ..graph import Graph, LabelBatch
from ..label_tensor import LabelTensor
import torch


class PinaDatasetFactory:
    """
    Factory class for the PINA dataset.

    Depending on the data type inside the conditions, it instanciate an object
    belonging to the appropriate subclass of 
    :class:`~pina.data.dataset.PinaDataset`. The possible subclasses are:

    - :class:`~pina.data.dataset.PinaTensorDataset`, for handling \
        :class:`torch.Tensor` and :class:`~pina.label_tensor.LabelTensor` data.
    - :class:`~pina.data.dataset.PinaGraphDataset`, for handling \
        :class:`~pina.graph.Graph` and :class:`~torch_geometric.data.Data` data.
    """

    def __new__(cls, conditions_dict, **kwargs):
        """
        Instantiate the appropriate subclass of
        :class:`~pina.data.dataset.PinaDataset`.

        If a graph is present in the conditions, returns a
        :class:`~pina.data.dataset.PinaGraphDataset`, otherwise returns a
        :class:`~pina.data.dataset.PinaTensorDataset`.

        :param dict conditions_dict: Dictionary containing all the conditions
            to be included in the dataset instance.
        :return: A subclass of :class:`~pina.data.dataset.PinaDataset`.
        :rtype: PinaTensorDataset | PinaGraphDataset

        :raises ValueError: If an empty dictionary is provided.
        """

        # Check if conditions_dict is empty
        if len(conditions_dict) == 0:
            raise ValueError("No conditions provided")

        dataset_dict = {}

        # Check is a Graph is present in the conditions
        for name, data in conditions_dict.items():
            if not isinstance(data, dict):
                raise ValueError(
                    f"Condition '{name}' data must be a dictionary"
                )

            # is_graph = cls._is_graph_dataset(conditions_dict)
            # if is_graph:
            #     raise NotImplementedError("PinaGraphDataset is not implemented yet.")

            dataset_dict[name] = PinaTensorDataset(data, **kwargs)
        return dataset_dict

    @staticmethod
    def _is_graph_dataset(cond_data):
        """
        TODO: Docstring
        """

        # Iterate over the values of the current condition
        for cond in cond_data.values():
            if isinstance(cond, (Data, Graph, list, tuple)):
                return True
        return False


class PinaTensorDataset(Dataset):
    """
    Dataset class for the PINA dataset with :class:`torch.Tensor` and
    :class:`~pina.label_tensor.LabelTensor` data.
    """

    def __init__(self, data_dict, automatic_batching=None):
        """
        Initialize the instance by storing the conditions dictionary.

        :param dict conditions_dict: A dictionary mapping condition names to
            their respective data. Each key represents a condition name, and the
            corresponding value is a dictionary containing the associated data.
        """

        # Store the conditions dictionary
        self.data = data_dict
        self.automatic_batching = (
            automatic_batching if automatic_batching is not None else True
        )
        self.stack_fn = (
            {}
        )  # LabelTensor.stack if any(isinstance(v, LabelTensor) for v in data_dict.values()) else torch.stack
        for k, v in data_dict.items():
            if isinstance(v, LabelTensor):
                self.stack_fn[k] = LabelTensor.stack
            elif isinstance(v, torch.Tensor):
                self.stack_fn[k] = torch.stack
            elif isinstance(v, list) and all(
                isinstance(item, (Data, Graph)) for item in v
            ):
                self.stack_fn[k] = LabelBatch.from_data_list
            else:
                raise ValueError(
                    f"Unsupported data type for stacking: {type(v)}"
                )

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        """
        Return the data at the given index in the dataset.

        :param int idx: Index.
        :return: A dictionary containing the data at the given index.
        :rtype: dict
        """

        if self.automatic_batching:
            # Return the data at the given index
            return {
                field_name: data[idx] for field_name, data in self.data.items()
            }
        return idx

    def _getitem_from_list(self, idx_list):
        """
        Return data from the dataset given a list of indices.

        :param list[int] idx_list: List of indices.
        :return: A dictionary containing the data at the given indices.
        :rtype: dict
        """

        to_return = {}
        for field_name, data in self.data.items():
            if self.stack_fn[field_name] == LabelBatch.from_data_list:
                to_return[field_name] = self.stack_fn[field_name](
                    [data[i] for i in idx_list]
                )
            else:
                to_return[field_name] = data[idx_list]
        return to_return


class PinaGraphDataset(Dataset):
    def __init__(self, data_dict, automatic_batching=None):
        """
        Initialize the instance by storing the conditions dictionary.

        :param dict conditions_dict: A dictionary mapping condition names to
            their respective data. Each key represents a condition name, and the
            corresponding value is a dictionary containing the associated data.
        """

        # Store the conditions dictionary
        self.data = data_dict
        self.automatic_batching = (
            automatic_batching if automatic_batching is not None else True
        )

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        """
        Return the data at the given index in the dataset.

        :param int idx: Index.
        :return: A dictionary containing the data at the given index.
        :rtype: dict
        """

        if self.automatic_batching:
            # Return the data at the given index
            return {
                field_name: data[idx] for field_name, data in self.data.items()
            }
        return idx

    def _getitem_from_list(self, idx_list):
        """
        Return data from the dataset given a list of indices.

        :param list[int] idx_list: List of indices.
        :return: A dictionary containing the data at the given indices.
        :rtype: dict
        """

        return {
            field_name: [data[i] for i in idx_list]
            for field_name, data in self.data.items()
        }
