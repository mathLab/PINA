"""
Batch management module
"""
from .pina_subset import PinaSubset


class Batch:
    """
    Implementation of the Batch class used during training to perform SGD
    optimization.
    """

    def __init__(self, dataset_dict, idx_dict, require_grad=True):
        self.attributes = []
        for k, v in dataset_dict.items():
            setattr(self, k, v)
            self.attributes.append(k)

        for k, v in idx_dict.items():
            setattr(self, k + '_idx', v)
        self.require_grad = require_grad

    def __len__(self):
        """
        Returns the number of elements in the batch
        :return: number of elements in the batch
        :rtype: int
        """
        length = 0
        for dataset in dir(self):
            attribute = getattr(self, dataset)
            if isinstance(attribute, list):
                length += len(getattr(self, dataset))
        return length

    def __getattribute__(self, item):
        if item in super().__getattribute__('attributes'):
            dataset = super().__getattribute__(item)
            index = super().__getattribute__(item + '_idx')
            return PinaSubset(
                dataset.dataset,
                dataset.indices[index])
        else:
            return super().__getattribute__(item)

    def __getattr__(self, item):
        if item == 'data' and len(self.attributes) == 1:
            item = self.attributes[0]
            return super().__getattribute__(item)
        raise AttributeError(f"'Batch' object has no attribute '{item}'")