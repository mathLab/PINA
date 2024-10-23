"""
This module is used to create an iterable object used during training
"""
import math
from .pina_batch import Batch


class PinaDataLoader:
    """
    This class is used to create a dataloader to use during the training.

    :var condition_names: The names of the conditions. The order is consistent
        with the condition indeces in the batches.
    :vartype condition_names: list[str]
    """

    def __init__(self, dataset_dict, batch_size, condition_names) -> None:
        """
        Initialize local variables
        :param dataset_dict: Dictionary of datasets
        :type dataset_dict: dict
        :param batch_size: Size of the batch
        :type batch_size: int
        :param condition_names: Names of the conditions
        :type condition_names: list[str]
        """
        self.condition_names = condition_names
        self.dataset_dict = dataset_dict
        self._init_batches(batch_size)

    def _init_batches(self, batch_size=None):
        """
        Create batches according to the batch_size provided in input.
        """
        self.batches = []
        n_elements = sum(len(v) for v in self.dataset_dict.values())
        if batch_size is None:
            batch_size = n_elements
        indexes_dict = {}
        n_batches = int(math.ceil(n_elements / batch_size))
        for k, v in self.dataset_dict.items():
            if n_batches != 1:
                indexes_dict[k] = math.floor(len(v) / (n_batches - 1))
            else:
                indexes_dict[k] = len(v)
        for i in range(n_batches):
            temp_dict = {}
            for k, v in indexes_dict.items():
                if i != n_batches - 1:
                    temp_dict[k] = slice(i * v, (i + 1) * v)
                else:
                    temp_dict[k] = slice(i * v, len(self.dataset_dict[k]))
            self.batches.append(
                Batch(idx_dict=temp_dict, dataset_dict=self.dataset_dict))

    def __iter__(self):
        """
        Makes dataloader object iterable
        """
        yield from self.batches

    def __len__(self):
        """
        Return the number of batches.
        :return: The number of batches.
        :rtype: int
        """
        return len(self.batches)
