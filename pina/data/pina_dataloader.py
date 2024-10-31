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
        self.batch_size = batch_size
        self._init_batches(batch_size)

    def _init_batches(self, batch_size=None):
        """
        Create batches according to the batch_size provided in input.
        """
        self.batches = []
        n_elements = sum(len(v) for v in self.dataset_dict.values())
        if batch_size is None:
            batch_size = n_elements
            self.batch_size = n_elements
        n_batches = int(math.ceil(n_elements / batch_size))
        indexes_dict = {
            k: math.floor(len(v) / n_batches) if n_batches != 1 else len(v) for
            k, v in self.dataset_dict.items()}

        dataset_names = list(self.dataset_dict.keys())
        num_el_per_batch = [{i: indexes_dict[i] for i in dataset_names} for _
                            in range(n_batches - 1)] + [
                               {i: 0 for i in dataset_names}]
        reminders = {
            i: len(self.dataset_dict[i]) - indexes_dict[i] * (n_batches - 1) for
            i in dataset_names}
        dataset_names = iter(dataset_names)
        name = next(dataset_names, None)
        for batch in num_el_per_batch:
            tot_num_el = sum(batch.values())
            batch_reminder = batch_size - tot_num_el
            for _ in range(batch_reminder):
                if name is None:
                    break
                if reminders[name] > 0:
                    batch[name] += 1
                    reminders[name] -= 1
                else:
                    name = next(dataset_names, None)
                    if name is None:
                        break
                    batch[name] += 1
                    reminders[name] -= 1

        reminders, dataset_names, indexes_dict = None, None, None  # free memory
        actual_indices = {k: 0 for k in self.dataset_dict.keys()}
        for batch in num_el_per_batch:
            temp_dict = {}
            total_length = 0
            for k, v in batch.items():
                temp_dict[k] = slice(actual_indices[k], actual_indices[k] + v)
                actual_indices[k] = actual_indices[k] + v
                total_length += v
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
