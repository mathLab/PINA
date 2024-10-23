"""
Basic data module implementation
"""
from torch.utils.data import Dataset
import torch
from ..label_tensor import LabelTensor
from ..graph import Graph


class BaseDataset(Dataset):
    """
    BaseDataset class, which handle initialization and data retrieval
    :var condition_indices: List of indices
    :var device: torch.device
    :var condition_names: dict of condition index and corresponding name
    """

    def __new__(cls, problem, device):
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

    def __init__(self, problem, device):
        """"
        Initialize the object based on __slots__
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.device device: The device on which the
        dataset will be loaded.
        """
        super().__init__()

        self.condition_names = {}
        collector = problem.collector
        for slot in self.__slots__:
            setattr(self, slot, [])
        num_el_per_condition = []
        idx = 0
        for name, data in collector.data_collections.items():
            keys = list(data.keys())
            current_cond_num_el = None
            if sorted(self.__slots__) == sorted(keys):
                for slot in self.__slots__:
                    slot_data = data[slot]
                    if isinstance(slot_data, (LabelTensor, torch.Tensor,
                                              Graph)):
                        if current_cond_num_el is None:
                            current_cond_num_el = len(slot_data)
                        elif current_cond_num_el != len(slot_data):
                            raise ValueError('Different number of conditions')
                    current_list = getattr(self, slot)
                    current_list += [data[slot]] if not (
                        isinstance(data[slot], list)) else data[slot]
                num_el_per_condition.append(current_cond_num_el)
                self.condition_names[idx] = name
                idx += 1
        if num_el_per_condition:
            self.condition_indices = torch.cat(
                [
                    torch.tensor([i] * num_el_per_condition[i],
                                 dtype=torch.uint8)
                    for i in range(len(num_el_per_condition))
                ],
                dim=0,
            )
            for slot in self.__slots__:
                current_attribute = getattr(self, slot)
                if all(isinstance(a, LabelTensor) for a in current_attribute):
                    setattr(self, slot, LabelTensor.vstack(current_attribute))
        else:
            self.condition_indices = torch.tensor([], dtype=torch.uint8)
            for slot in self.__slots__:
                setattr(self, slot, torch.tensor([]))
        self.device = device

    def __len__(self):
        return len(getattr(self, self.__slots__[0]))

    def __getattribute__(self, item):
        attribute = super().__getattribute__(item)
        if isinstance(attribute,
                      LabelTensor) and attribute.dtype == torch.float32:
            attribute = attribute.to(device=self.device).requires_grad_()
        return attribute

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return getattr(self, idx).to(self.device)
        if isinstance(idx, slice):
            to_return_list = []
            for i in self.__slots__:
                to_return_list.append(getattr(self, i)[idx].to(self.device))
            return to_return_list

        if isinstance(idx, (tuple, list)):
            if (len(idx) == 2 and isinstance(idx[0], str)
                    and isinstance(idx[1], (list, slice))):
                tensor = getattr(self, idx[0])
                return tensor[[idx[1]]].to(self.device)
            if all(isinstance(x, int) for x in idx):
                to_return_list = []
                for i in self.__slots__:
                    to_return_list.append(
                        getattr(self, i)[[idx]].to(self.device))
                return to_return_list

        raise ValueError(f'Invalid index {idx}')
