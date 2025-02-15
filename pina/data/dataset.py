"""
This module provide basic data management functionalities
"""
import torch
from torch.utils.data import Dataset
from abc import abstractmethod
from torch_geometric.data import Batch


class PinaDatasetFactory:
    """
    Factory class for the PINA dataset. Depending on the type inside the
    conditions it creates a different dataset object:
    - PinaTensorDataset for torch.Tensor
    - PinaGraphDataset for list of torch_geometric.data.Data objects
    """

    def __new__(cls, conditions_dict, **kwargs):
        if len(conditions_dict) == 0:
            raise ValueError('No conditions provided')
        if all([isinstance(v['input_points'], torch.Tensor) for v
                in conditions_dict.values()]):
            return PinaTensorDataset(conditions_dict, **kwargs)
        elif all([isinstance(v['input_points'], list) for v
                  in conditions_dict.values()]):
            return PinaGraphDataset(conditions_dict, **kwargs)
        raise ValueError('Conditions must be either torch.Tensor or list of Data '
                         'objects.')


class PinaDataset(Dataset):
    """
    Abstract class for the PINA dataset
    """

    def __init__(self, conditions_dict, max_conditions_lengths):
        self.conditions_dict = conditions_dict
        self.max_conditions_lengths = max_conditions_lengths
        self.conditions_length = {k: len(v['input_points']) for k, v in
                                  self.conditions_dict.items()}
        self.length = max(self.conditions_length.values())

    def _get_max_len(self):
        max_len = 0
        for condition in self.conditions_dict.values():
            max_len = max(max_len, len(condition['input_points']))
        return max_len

    def __len__(self):
        return self.length

    @abstractmethod
    def __getitem__(self, item):
        pass


class PinaTensorDataset(PinaDataset):
    def __init__(self, conditions_dict, max_conditions_lengths,
                 automatic_batching):
        super().__init__(conditions_dict, max_conditions_lengths)

        if automatic_batching:
            self._getitem_func = self._getitem_int
        else:
            self._getitem_func = self._getitem_list

    def _getitem_int(self, idx):
        return {
            k: {k_data: v[k_data][idx % len(v['input_points'])] for k_data
                in v.keys()} for k, v in self.conditions_dict.items()
        }

    def fetch_from_idx_list(self, idx):
        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            cond_idx = idx[:self.max_conditions_lengths[condition]]
            condition_len = self.conditions_length[condition]
            if self.length > condition_len:
                cond_idx = [idx % condition_len for idx in cond_idx]
            to_return_dict[condition] = {k: v[cond_idx]
                                         for k, v in data.items()}
        return to_return_dict

    @staticmethod
    def _getitem_list(idx):
        return idx

    def get_all_data(self):
        index = [i for i in range(len(self))]
        return self.fetch_from_idx_list(index)

    def __getitem__(self, idx):
        return self._getitem_func(idx)


class PinaGraphDataset(PinaDataset):

    def __init__(self, conditions_dict, max_conditions_lengths,
                 automatic_batching):
        super().__init__(conditions_dict, max_conditions_lengths)
        if automatic_batching:
            self._getitem_func = self._getitem_int
        else:
            self._getitem_func = self._getitem_list

    def fetch_from_idx_list(self, idx):
        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            cond_idx = idx[:self.max_conditions_lengths[condition]]
            condition_len = self.conditions_length[condition]
            if self.length > condition_len:
                cond_idx = [idx % condition_len for idx in cond_idx]
            to_return_dict[condition] = {k: Batch.from_data_list([v[i]
                                                                  for i in cond_idx])
            if isinstance(v, list)
            else v[cond_idx].reshape(-1, *v[cond_idx].shape[2:])
                                         for k, v in data.items()
                                         }
        return to_return_dict

    def _getitem_list(self, idx):
        return idx

    def _getitem_int(self, idx):
        return {
            k: {k_data: v[k_data][idx % len(v['input_points'])] for k_data
                in v.keys()} for k, v in self.conditions_dict.items()
        }

    def get_all_data(self):
        index = [i for i in range(len(self))]
        return self.fetch_from_idx_list(index)

    def __getitem__(self, idx):
        return self._getitem_func(idx)