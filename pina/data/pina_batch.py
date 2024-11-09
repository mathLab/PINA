"""
Batch management module
"""
from attr import attributes

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
