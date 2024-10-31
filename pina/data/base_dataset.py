"""
Basic data module implementation
"""
import torch
import logging

from torch.utils.data import Dataset

from ..label_tensor import LabelTensor


class BaseDataset(Dataset):
    """
    BaseDataset class, which handle initialization and data retrieval
    :var condition_indices: List of indices
    :var device: torch.device
    """

    def __new__(cls, problem=None, device=torch.device('cpu')):
        """
        Ensure correct definition of __slots__ before initialization
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.device device: The device on which the
        dataset will be loaded.
        """
        if cls is BaseDataset:
            raise TypeError(
                'BaseDataset cannot be instantiated directly. Use a subclass.')
        if not hasattr(cls, '__slots__'):
            raise TypeError(
                'Something is wrong, __slots__ must be defined in subclasses.')
        return object.__new__(cls)

    def __init__(self, problem=None, device=torch.device('cpu')):
        """"
        Initialize the object based on __slots__
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.device device: The device on which the
        dataset will be loaded.
        """
        super().__init__()
        self.empty = True
        self.problem = problem
        self.device = device
        self.condition_indices = None
        for slot in self.__slots__:
            setattr(self, slot, [])
        self.num_el_per_condition = []
        self.conditions_idx = []
        if self.problem is not None:
            self._init_from_problem(self.problem.collector.data_collections)
        self.initialized = False

    def _init_from_problem(self, collector_dict):
        """
        TODO
        """
        for name, data in collector_dict.items():
            keys = list(data.keys())
            if set(self.__slots__) == set(keys):
                self._populate_init_list(data)
                idx = [
                    key for key, val in
                    self.problem.collector.conditions_name.items()
                    if val == name
                ]
                self.conditions_idx.append(idx)
        self.initialize()

    def add_points(self, data_dict, condition_idx, batching_dim=0):
        """
        This method filled internal lists of data points
        :param data_dict: dictionary containing data points
        :param condition_idx: index of the condition to which the data points
                belong to
        :param batching_dim: dimension of the batching
        :raises: ValueError if the dataset has already been initialized
        """
        if not self.initialized:
            self._populate_init_list(data_dict, batching_dim)
            self.conditions_idx.append(condition_idx)
            self.empty = False
        else:
            raise ValueError('Dataset already initialized')

    def _populate_init_list(self, data_dict, batching_dim=0):
        current_cond_num_el = None
        for slot in data_dict.keys():
            slot_data = data_dict[slot]
            if batching_dim != 0:
                if isinstance(slot_data, (LabelTensor, torch.Tensor)):
                    dims = len(slot_data.size())
                    slot_data = slot_data.permute(
                        [batching_dim] +
                        [dim for dim in range(dims) if dim != batching_dim])
            if current_cond_num_el is None:
                current_cond_num_el = len(slot_data)
            elif current_cond_num_el != len(slot_data):
                raise ValueError('Different dimension in same condition')
            current_list = getattr(self, slot)
            current_list += [
                slot_data
            ] if not (isinstance(slot_data, list)) else slot_data
        self.num_el_per_condition.append(current_cond_num_el)

    def initialize(self):
        """
        Initialize the datasets tensors/LabelTensors/lists given the lists
        already filled
        """
        logging.debug(f'Initialize dataset {self.__class__.__name__}')

        if self.num_el_per_condition:
            self.condition_indices = torch.cat([
                torch.tensor([i] * self.num_el_per_condition[i],
                             dtype=torch.uint8)
                for i in range(len(self.num_el_per_condition))
            ],
                                               dim=0)
            for slot in self.__slots__:
                current_attribute = getattr(self, slot)
                if all(isinstance(a, LabelTensor) for a in current_attribute):
                    setattr(self, slot, LabelTensor.vstack(current_attribute))
        self.initialized = True

    def __len__(self):
        """
        :return: Number of elements in the dataset
        """
        return len(getattr(self, self.__slots__[0]))

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        if not isinstance(idx, (tuple, list, slice, int)):
            raise IndexError("Invalid index")
        tensors = []
        for attribute in self.__slots__:
            tensor = getattr(self, attribute)
            if isinstance(attribute, (LabelTensor, torch.Tensor)):
                tensors.append(tensor.__getitem__(idx))
            elif isinstance(attribute, list):
                if isinstance(idx, (list, tuple)):
                    tensor = [tensor[i] for i in idx]
                tensors.append(tensor)
        return tensors

    def apply_shuffle(self, indices):
        for slot in self.__slots__:
            if slot != 'equation':
                attribute = getattr(self, slot)
                if isinstance(attribute, (LabelTensor, torch.Tensor)):
                    setattr(self, 'slot', attribute[[indices]])
                if isinstance(attribute, list):
                    setattr(self, 'slot', [attribute[i] for i in indices])
