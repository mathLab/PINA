"""
Batch management module
"""
import torch
from ..label_tensor import LabelTensor

from .pina_subset import PinaSubset


class Batch:
    """
    Implementation of the Batch class used during training to perform SGD
    optimization.
    """

    def __init__(self, dataset_dict, idx_dict, require_grad=True):
        self.attributes = []
        for k, v in dataset_dict.items():
            index = idx_dict[k]
            if isinstance(v, PinaSubset):
                dataset_index = v.indices
                if isinstance(dataset_index, slice):
                    index = slice(dataset_index.start + index.start,
                                  min(dataset_index.start + index.stop,
                                      dataset_index.stop))
            setattr(self, k, PinaSubset(v.dataset, index,
                                        require_grad=require_grad))
            self.attributes.append(k)
        self.require_grad = require_grad

    def __len__(self):
        """
        Returns the number of elements in the batch
        :return: number of elements in the batch
        :rtype: int
        """
        length = 0
        for dataset in self.attributes:
            attribute = getattr(self, dataset)
            length += len(attribute)
        return length

    def __getattr__(self, item):
        if item == 'data' and len(self.attributes) == 1:
            item = self.attributes[0]
            return self.__getattribute__(item)
        raise AttributeError(f"'Batch' object has no attribute '{item}'")

    def get_data(self, batch_name=None):
        """
        # TODO
        """
        data = getattr(self, batch_name)
        to_return_list = []
        if isinstance(data, PinaSubset):
            items = data.dataset.__slots__
        else:
            items = data.__slots__
        indices = torch.unique(data.condition_indices).tolist()
        condition_idx = data.condition_indices
        for i in indices:
            temp = []
            for j in items:
                var = getattr(data, j)
                if isinstance(var, (torch.Tensor, LabelTensor)):
                    temp.append(var[i == condition_idx])
                if isinstance(var, list) and len(var) > 0:
                    temp.append([var[k] for k in range(len(var)) if
                                 i == condition_idx[k]])
            temp.append(i)
            to_return_list.append(temp)
        return to_return_list

    def get_supervised_data(self):
        """
        Get a subset of the batch
        :param idx: indices of the subset
        :type idx: slice
        :return: subset of the batch
        :rtype: Batch
        """
        return self.get_data(batch_name='supervised')

    def get_physics_data(self):
        """
        Get a subset of the batch
        :param idx: indices of the subset
        :type idx: slice
        :return: subset of the batch
        :rtype: Batch
        """
        return self.get_data(batch_name='physics')
