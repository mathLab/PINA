"""
This module provide basic data management functionalities
"""

import functools
import torch
from torch.utils.data import Dataset
from abc import abstractmethod
from torch_geometric.data import Batch, Data
from pina import LabelTensor


class PinaDatasetFactory:
    """
    Factory class for the PINA dataset. Depending on the type inside the
    conditions it creates a different dataset object:
    - PinaTensorDataset for torch.Tensor
    - PinaGraphDataset for list of torch_geometric.data.Data objects
    """

    def __new__(cls, conditions_dict, **kwargs):
        if len(conditions_dict) == 0:
            raise ValueError("No conditions provided")
        if all(
            [
                isinstance(v["input_points"], torch.Tensor)
                for v in conditions_dict.values()
            ]
        ):
            return PinaTensorDataset(conditions_dict, **kwargs)
        elif all(
            [
                isinstance(v["input_points"], list)
                for v in conditions_dict.values()
            ]
        ):
            return PinaGraphDataset(conditions_dict, **kwargs)
        raise ValueError(
            "Conditions must be either torch.Tensor or list of Data " "objects."
        )


class PinaDataset(Dataset):
    """
    Abstract class for the PINA dataset
    """

    def __init__(self, conditions_dict, max_conditions_lengths):
        self.conditions_dict = conditions_dict
        self.max_conditions_lengths = max_conditions_lengths
        self.conditions_length = {
            k: len(v["input_points"]) for k, v in self.conditions_dict.items()
        }
        self.length = max(self.conditions_length.values())

    def _get_max_len(self):
        max_len = 0
        for condition in self.conditions_dict.values():
            max_len = max(max_len, len(condition["input_points"]))
        return max_len

    def __len__(self):
        return self.length

    @abstractmethod
    def __getitem__(self, item):
        pass


class PinaTensorDataset(PinaDataset):
    def __init__(
        self, conditions_dict, max_conditions_lengths, automatic_batching
    ):
        super().__init__(conditions_dict, max_conditions_lengths)

        if automatic_batching:
            self._getitem_func = self._getitem_int
        else:
            self._getitem_func = self._getitem_dummy

    def _getitem_int(self, idx):
        return {
            k: {
                k_data: v[k_data][idx % len(v["input_points"])]
                for k_data in v.keys()
            }
            for k, v in self.conditions_dict.items()
        }

    def fetch_from_idx_list(self, idx):
        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            cond_idx = idx[: self.max_conditions_lengths[condition]]
            condition_len = self.conditions_length[condition]
            if self.length > condition_len:
                cond_idx = [idx % condition_len for idx in cond_idx]
            to_return_dict[condition] = {
                k: v[cond_idx] for k, v in data.items()
            }
        return to_return_dict

    @staticmethod
    def _getitem_dummy(idx):
        return idx

    def get_all_data(self):
        index = [i for i in range(len(self))]
        return self.fetch_from_idx_list(index)

    def __getitem__(self, idx):
        return self._getitem_func(idx)

    @property
    def input_points(self):
        """
        Method to return input points for training.
        """
        return {k: v["input_points"] for k, v in self.conditions_dict.items()}


class PinaBatch(Batch):
    """
    Add extract function to torch_geometric Batch object
    """

    def __init__(self):

        super().__init__(self)

    def extract(self, labels):
        """
        Perform extraction of labels on node features (x)

        :param labels: Labels to extract
        :type labels: list[str] | tuple[str] | str
        :return: Batch object with extraction performed on x
        :rtype: PinaBatch
        """
        self.x = self.x.extract(labels)
        return self


class PinaGraphDataset(PinaDataset):

    def __init__(
        self, conditions_dict, max_conditions_lengths, automatic_batching
    ):
        super().__init__(conditions_dict, max_conditions_lengths)
        self.in_labels = {}
        self.out_labels = None
        if automatic_batching:
            self._getitem_func = self._getitem_int
        else:
            self._getitem_func = self._getitem_dummy

        ex_data = conditions_dict[list(conditions_dict.keys())[0]][
            "input_points"
        ][0]
        for name, attr in ex_data.items():
            if isinstance(attr, LabelTensor):
                self.in_labels[name] = attr.stored_labels
        ex_data = conditions_dict[list(conditions_dict.keys())[0]][
            "output_points"
        ][0]
        if isinstance(ex_data, LabelTensor):
            self.out_labels = ex_data.labels

        self._create_graph_batch_from_list = (
            self._labelise_batch(self._base_create_graph_batch_from_list)
            if self.in_labels
            else self._base_create_graph_batch_from_list
        )

        self._create_output_batch = (
            self._labelise_tensor(self._base_create_output_batch)
            if self.out_labels is not None
            else self._base_create_output_batch
        )

    def fetch_from_idx_list(self, idx):
        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            cond_idx = idx[: self.max_conditions_lengths[condition]]
            condition_len = self.conditions_length[condition]
            if self.length > condition_len:
                cond_idx = [idx % condition_len for idx in cond_idx]
            to_return_dict[condition] = {
                k: (
                    self._create_graph_batch_from_list([v[i] for i in idx])
                    if isinstance(v, list)
                    else self._create_output_batch(v[idx])
                )
                for k, v in data.items()
            }

        return to_return_dict

    def _base_create_graph_batch_from_list(self, data):
        batch = PinaBatch.from_data_list(data)
        return batch

    def _base_create_output_batch(self, data):
        out = data.reshape(-1, *data.shape[2:])
        return out

    def _getitem_dummy(self, idx):
        return idx

    def _getitem_int(self, idx):
        return {
            k: {
                k_data: v[k_data][idx % len(v["input_points"])]
                for k_data in v.keys()
            }
            for k, v in self.conditions_dict.items()
        }

    def get_all_data(self):
        index = [i for i in range(len(self))]
        return self.fetch_from_idx_list(index)

    def __getitem__(self, idx):
        return self._getitem_func(idx)

    def _labelise_batch(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            batch = func(*args, **kwargs)
            for k, v in self.in_labels.items():
                tmp = batch[k]
                tmp.labels = v
                batch[k] = tmp
            return batch

        return wrapper

    def _labelise_tensor(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            if isinstance(out, LabelTensor):
                out.labels = self.out_labels
            return out

        return wrapper

    def create_graph_batch(self, data):
        """
        # TODO
        """
        if isinstance(data[0], Data):
            return self._create_graph_batch_from_list(data)
        return self._create_output_batch(data)
