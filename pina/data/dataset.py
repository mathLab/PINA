"""Module for the PINA dataset classes."""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ..graph import Graph, LabelBatch
from ..label_tensor import LabelTensor


class PinaDatasetFactory:
    """
    Factory class to create PINA datasets based on the provided conditions
    dictionary.
    :param dict conditions_dict: A dictionary where keys are condition names
        and values are dictionaries containing the associated data.
    :return: A dictionary mapping condition names to their respective
        :class:`PinaDataset` instances.
    """

    def __new__(cls, conditions_dict, **kwargs):
        """
        Create PINA dataset instances based on the provided conditions
        dictionary.
        :param dict conditions_dict: A dictionary where keys are condition names
            and values are dictionaries containing the associated data.
        :return: A dictionary mapping condition names to their respective
            :class:`PinaDataset` instances.
        """

        # Check if conditions_dict is empty
        if len(conditions_dict) == 0:
            raise ValueError("No conditions provided")

        dataset_dict = {}  # Dictionary to hold the created datasets

        # Check is a Graph is present in the conditions
        for name, data in conditions_dict.items():
            # Validate that data is a dictionary
            if not isinstance(data, dict):
                raise ValueError(
                    f"Condition '{name}' data must be a dictionary"
                )
            # Create PinaDataset instance for each condition
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
        self.is_graph_dataset = False
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
                self.is_graph_dataset = True
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

    def getitem_from_list(self, idx_list):
        """
        Return data from the dataset given a list of indices.

        :param list[int] idx_list: List of indices.
        :return: A dictionary containing the data at the given indices.
        :rtype: dict
        """

        to_return = {}
        for field_name, data in self.data.items():
            if self.stack_fn[field_name] is LabelBatch.from_data_list:
                to_return[field_name] = self.stack_fn[field_name](
                    [data[i] for i in idx_list]
                )
            else:
                to_return[field_name] = data[idx_list]
        return to_return

    def update_data(self, update_dict):
        """
        Update the dataset's data in-place.
        :param dict update_dict: A dictionary where keys are condition names
            and values are dictionaries with updated data for those conditions.
        """
        for field_name, updates in update_dict.items():
            if field_name not in self.data:
                raise KeyError(
                    f"Condition '{field_name}' not found in dataset."
                )
            if not isinstance(updates, (LabelTensor, torch.Tensor)):
                raise ValueError(
                    f"Updates for condition '{field_name}' must be of type "
                    f"LabelTensor or torch.Tensor."
                )
            self.data[field_name] = updates

    @property
    def input(self):
        """
        Get the input data from the dataset.

        :return: The input data.
        :rtype: torch.Tensor | LabelTensor | Data | Graph
        """
        return self.data["input"]
