"""Module for the PINA dataset classes."""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ..graph import Graph, LabelBatch
from ..label_tensor import LabelTensor


class PinaDatasetFactory:
    """
    TODO: Update docstring
    """

    def __new__(cls, conditions_dict, **kwargs):
        """
        TODO: Update docstring
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
            dataset_dict[name] = PinaDataset(data, **kwargs)
        return dataset_dict


class PinaDataset(Dataset):
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
        self.stack_fn = {}
        # Determine stacking functions for each data type (used in collate_fn)
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
