"""
Batch management module
"""
from .pina_subset import PinaSubset


class Batch:
    """
    Implementation of the Batch class used during training to perform SGD optimization.
    """

    def __init__(self, dataset_dict, idx_dict):

        for k, v in dataset_dict.items():
            setattr(self, k, v)

        for k, v in idx_dict.items():
            setattr(self, k + '_idx', v)

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

    def __getattr__(self, item):
        if not item in dir(self):
            raise AttributeError(f'Batch instance has no attribute {item}')
        return PinaSubset(
            getattr(self, item).dataset,
            getattr(self, item).indices[self.coordinates_dict[item]])
