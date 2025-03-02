"""
This module provide basic data management functionalities
"""

import functools
from abc import abstractmethod
from torch.utils.data import Dataset
from torch_geometric.data import Batch
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
        print(conditions_dict)
        if all("graph" in list(v.keys()) for v in conditions_dict.values()):
            return PinaGraphDataset(conditions_dict, **kwargs)
        if all(
            "input_points" in list(v.keys()) for v in conditions_dict.values()
        ):
            return PinaTensorDataset(conditions_dict, **kwargs)
        raise ValueError(
            "Conditions must be either torch.Tensor or list of Data objects."
        )


class PinaDataset(Dataset):
    """
    Abstract class for the PINA dataset
    """

    def __init__(self, conditions_dict, max_conditions_lengths):
        self.conditions_dict = conditions_dict
        self.max_conditions_lengths = max_conditions_lengths

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

    def get_all_data(self):
        """
        Get all the data from the dataset

        :return: dictionary with the data for each condition
        :rtype: dict
        """
        index = list(range(len(self)))
        return self.fetch_from_idx_list(index)


class PinaTensorDataset(PinaDataset):
    """
    Dataset class for torch.Tensor conditions
    """

    def __init__(
        self, conditions_dict, max_conditions_lengths, automatic_batching
    ):
        """
        Initialize the dataset, assign the conditions and maximum lengths
        for each condition. Moreover, it sets the right function to get
        the data from the dataset.

        :param dict conditions_dict: dictionary with conditions
        :param dict max_conditions_lengths: maximum length of each condition
        :param bool automatic_batching: if True, the dataset will return
            a single condition for each index, otherwise it will return the
            index itself
        """
        super().__init__(conditions_dict, max_conditions_lengths)
        self.conditions_length = {
            k: len(v["input_points"]) for k, v in self.conditions_dict.items()
        }
        self.length = max(self.conditions_length.values())
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
        """
        Retrive data from the dataset given a list of indexes.

        :param list idx: list of indexes
        :return: dictionary with the data for each condition
        :rtype: dict
        """
        # Set empty dictionary
        to_return_dict = {}
        # Loop over conditions
        for condition, data in self.conditions_dict.items():
            # Get the indexes for the current condition
            cond_idx = idx[: self.max_conditions_lengths[condition]]
            # Get the length of the current condition
            condition_len = self.conditions_length[condition]
            # If the length of the dataset is greater than the length of the
            # condition, we need to take the modulo of the index
            if self.length > condition_len:
                cond_idx = [idx % condition_len for idx in cond_idx]
            # Store the data for the current condition
            to_return_dict[condition] = {
                k: v[cond_idx] for k, v in data.items()
            }
        return to_return_dict

    @staticmethod
    def _getitem_dummy(idx):
        return idx

    def __getitem__(self, idx):
        return self._getitem_func(idx)

    @property
    def input_points(self):
        """
        Method to return input points for training.
        """
        return {k: v["input_points"] for k, v in self.conditions_dict.items()}


class PinaGraphDataset(PinaDataset):
    """
    Dataset class for torch_geometric.data.Data and Graph conditions
    """

    def __init__(
        self, conditions_dict, max_conditions_lengths, automatic_batching
    ):
        """
        Initialize the dataset, assign the conditions and maximum lengths
        for each condition. Moreover, it sets the right function to get
        the data from the dataset.

        :param dict conditions_dict: dictionary with conditions
        :param dict max_conditions_lengths: maximum length of each condition
        :param bool automatic_batching: if True, the dataset will return
            a single condition for each index, otherwise it will return the
            index itself
        """
        super().__init__(conditions_dict, max_conditions_lengths)
        self.conditions_length = {
            k: len(v["graph"]) for k, v in self.conditions_dict.items()
        }
        self.length = max(self.conditions_length.values())

        self.in_labels = {}
        self.out_labels = None
        if automatic_batching:
            self._getitem_func = self._getitem_int
        else:
            self._getitem_func = self._getitem_dummy

        ex_data = conditions_dict[list(conditions_dict.keys())[0]]["graph"][0]

        for name, attr in ex_data.items():
            if isinstance(attr, LabelTensor):
                self.in_labels[name] = attr.stored_labels

        self._create_graph_batch_from_list = (
            self._labelise_batch(self._base_create_graph_batch_from_list)
            if self.in_labels
            else self._base_create_graph_batch_from_list
        )
        if hasattr(ex_data, "y"):
            self.divide_batch = self._extract_output(self._divide_batch)
        else:
            self.divide_batch = self._divide_batch

    def fetch_from_idx_list(self, idx):
        """
        Retrive data from the dataset given a list of indexes.

        :param list idx: list of indexes
        :return: dictionary with the data for each condition
        :rtype dict
        """
        to_return_dict = {}
        for condition, data in self.conditions_dict.items():
            cond_idx = idx[: self.max_conditions_lengths[condition]]
            condition_len = self.conditions_length[condition]
            if self.length > condition_len:
                cond_idx = [idx % condition_len for idx in cond_idx]
            batch = self._create_graph_batch_from_list(
                [data["graph"][i] for i in idx]
            )
            to_return_dict[condition] = self.divide_batch(batch=batch)
        return to_return_dict

    def _divide_batch(self, batch):
        """
        Divide the batch into input and output points
        """
        to_return_dict = {}
        to_return_dict["input_points"] = batch
        return to_return_dict

    def _base_create_graph_batch_from_list(self, data):
        batch = Batch.from_data_list(data)
        return batch

    def _getitem_dummy(self, idx):
        return idx

    def _getitem_int(self, idx):
        return {
            k: v["graph"][idx % len(v["graph"])]
            for k, v in self.conditions_dict.items()
        }

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
        Create a graph batch from a list of Data objects. This method is
        to be called from the outside.

        :param list data: list of Data or Graph objects
        :return: Batch object
        :rtype: torch_geometric.data.Batch
        """
        return self._create_graph_batch_from_list(data)

    @staticmethod
    def _extract_output(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            batch = kwargs["batch"]
            # Copying y into ouput_points
            out["output_points"] = batch.y
            # Deleting y from batch
            batch.y = None
            # Store new batch withou y
            out["input_points"] = batch
            return out

        return wrapper

    @staticmethod
    def _extract_cond_vars(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            batch = kwargs["batch"]
            # Copying conditional_vars into conditional_vars dict item
            out["conditional_variables"] = batch.conditional_vars
            # Deleting conditional_vars from batch
            batch.conditional_vars = None
            # Store new batch withou conditional_vars
            out["input_points"] = batch
            return out

        return wrapper
