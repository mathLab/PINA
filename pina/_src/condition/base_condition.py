"""Module for the Base Condition class."""

from functools import partial
import torch
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from pina._src.condition.condition_interface import ConditionInterface
from pina._src.core.graph import LabelBatch
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.utils import check_consistency
from pina._src.data.dummy_dataloader import DummyDataloader
from pina._src.problem.problem_interface import ProblemInterface


class BaseCondition(ConditionInterface):
    """
    Base class for all conditions, implementing common functionality.

    All specific condition types should inherit from this class and implement
    the abstract methods of
    :class:`~pina.condition.condition_interface.ConditionInterface`.

    This class is not meant to be instantiated directly.
    """

    # Available collate functions for automatic batching
    collate_fn_dict = {
        "tensor": torch.stack,
        "label_tensor": LabelTensor.stack,
        "graph": LabelBatch.from_data_list,
        "data": Batch.from_data_list,
    }

    def __init__(self, **kwargs):
        """
        Initialization of the :class:`BaseCondition` class.

        :param dict kwargs: The keyword arguments representing the data to be
            stored in the condition.
        """
        super().__init__()
        self.data = self.store_data(**kwargs)
        self.has_custom_dataloader_fn = False

    def __len__(self):
        """
        Return the number of data points in the condition.

        :return: The number of data points.
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the data point at the specified index.

        :param int idx: The index of the data point to retrieve.
        :return: The data point at the specified index.
        :rtype: Any
        """
        return self.data[idx]

    def create_dataloader(
        self, dataset, batch_size, automatic_batching, **kwargs
    ):
        """
        Create the DataLoader for the condition.

        :param _ConditionSubset dataset: The dataset for the DataLoader.
        :param int batch_size: The batch size for the DataLoader.
        :param bool automatic_batching: Whether to use automatic batching.
        :param dict kwargs: Additional keyword arguments for the DataLoader.
        :return: The DataLoader for the condition.
        :rtype: torch.utils.data.DataLoader
        """
        # If batching the entire dataset, return a DummyDataloader
        if batch_size == len(dataset):
            return DummyDataloader(dataset)

        # Otherwise, return a regular DataLoader with the appropriate collate
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

    def switch_dataloader_fn(self, create_dataloader_fn):
        """
        Switch the dataloader function for the condition.

        :param Callable create_dataloader_fn: The new dataloader function to use
            for the condition.
        :return: The new dataloader function for the condition.
        :rtype: Callable
        """
        self.has_custom_dataloader_fn = True
        self.create_dataloader = create_dataloader_fn

    @classmethod
    def automatic_batching_collate_fn(cls, batch):
        """
        Collate function for automatic batching to be used in the DataLoader.

        :param list batch: A list of items from the dataset.
        :return: A collated batch.
        :rtype: dict
        """
        # If the batch is empty, return an empty dictionary
        if not batch:
            return {}

        # Otherwise, collate the batch using the appropriate collate function
        instance_class = batch[0].__class__
        return instance_class.create_batch(batch)

    @staticmethod
    def collate_fn(batch, condition):
        """
        Collate function for custom batching to be used in the DataLoader.

        :param list batch: A list of items from the dataset.
        :param BaseCondition condition: The condition instance.
        :return: A collated batch.
        :rtype: dict
        """
        return condition.data[batch].to_batch()

    @property
    def problem(self):
        """
        The problem associated with this condition.

        :return: The problem associated with this condition.
        :rtype: BaseProblem
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param BaseProblem value: The problem to associate with this condition.
        :raises ValueError: If the problem is not an instance of BaseProblem.
        """
        check_consistency(value, ProblemInterface)
        self._problem = value
