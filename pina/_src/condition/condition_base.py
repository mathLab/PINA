"""
Base class for conditions.
"""

from functools import partial
import torch
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from pina._src.condition.condition_interface import ConditionInterface
from pina._src.core.graph import LabelBatch
from pina._src.core.label_tensor import LabelTensor
from pina._src.data.dummy_dataloader import DummyDataloader


class ConditionBase(ConditionInterface):
    """
    Base abstract class for all conditions in PINA.
    This class provides common functionality for handling data storage,
    batching, and interaction with the associated problem.
    """

    collate_fn_dict = {
        "tensor": torch.stack,
        "label_tensor": LabelTensor.stack,
        "graph": LabelBatch.from_data_list,
        "data": Batch.from_data_list,
    }

    def __init__(self, **kwargs):
        """
        Initialization of the :class:`ConditionBase` class.

        :param kwargs: Keyword arguments representing the data to be stored.
        """
        super().__init__()
        self.data = self.store_data(**kwargs)

    @property
    def problem(self):
        """
        Return the problem associated with this condition.

        :return: Problem associated with this condition.
        :rtype: ~pina.problem.abstract_problem.AbstractProblem
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param pina.problem.abstract_problem.AbstractProblem value: The problem
            to associate with this condition
        """
        self._problem = value

    def __len__(self):
        """
        Return the number of data points in the condition.

        :return: Number of data points.
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the data point(s) at the specified index.

        :param idx: Index(es) of the data point(s) to retrieve.
        :type idx: int | list[int]
        :return: Data point(s) at the specified index.
        """
        return self.data[idx]

    @classmethod
    def automatic_batching_collate_fn(cls, batch):
        """
        Collate function for automatic batching to be used in DataLoader.
        :param batch: A list of items from the dataset.
        :type batch: list
        :return: A collated batch.
        :rtype: dict
        """
        if not batch:
            return {}
        instance_class = batch[0].__class__
        batch = instance_class.create_batch(batch)
        return batch

    @staticmethod
    def collate_fn(batch, condition):
        """
        Collate function for custom batching to be used in DataLoader.

        :param batch: A list of items from the dataset.
        :type batch: list
        :param condition: The condition instance.
        :type condition: ConditionBase
        :return: A collated batch.
        :rtype: dict
        """
        data = condition.data[batch].to_batch()
        return data

    def create_dataloader(
        self,
        dataset,
        batch_size,
        automatic_batching,
        **kwargs,
    ):
        """
        Create a DataLoader for the condition.

        :param int batch_size: The batch size for the DataLoader.
        :param bool shuffle: Whether to shuffle the data. Default is ``False``.
        :return: The DataLoader for the condition.
        :rtype: torch.utils.data.DataLoader
        """
        if batch_size == len(dataset):
            return DummyDataloader(dataset)
        return DataLoader(
            dataset=dataset,
            collate_fn=(
                partial(self.collate_fn, condition=self)
                if not automatic_batching
                else self.automatic_batching_collate_fn
            ),
            batch_size=batch_size,
            **kwargs,
        )
