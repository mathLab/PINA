"""Module for the Tensor-Data Manager class."""

import torch
from pina._src.core.label_tensor import LabelTensor
from pina._src.condition.batch_manager import _BatchManager
from pina._src.condition.data_manager_interface import _DataManagerInterface


class _TensorDataManager(_DataManagerInterface):
    """
    Data manager for tensor-based data. It handles inputs stored as
    :class:`torch.Tensor` or :class:`~pina.label_tensor.LabelTensor`.
    """

    def __init__(self, **kwargs):
        """
        Initialization of the :class:`_TensorDataManager` class.

        :param dict kwargs: The keyword arguments for the tensor data manager.
        """
        self.keys = list(kwargs.keys())
        self.data = kwargs

        # Set attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        """
        Return the number of samples in the tensor data manager.

        :return: The number of samples.
        :rtype: int
        """
        return self.data[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        """
        Return the item at the specified indices.

        :param idx: The indices of the data point to retrieve.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: A new :class:`_TensorDataManager` instance containing the
            selected data items.
        :rtype: _TensorDataManager
        """
        # Get data at selected indices
        new_data = {
            k: (self.data[k][idx] if k in self.keys else self.data[k])
            for k in self.keys
        }

        return _TensorDataManager(**new_data)

    def to_batch(self):
        """
        Create a batch from the current tensor data manager.

        :return: A new instance of :class:`_BatchManager` with batched data.
        :rtype: _BatchManager
        """
        # Create the batch manager
        batch_data = _BatchManager()
        for k in self.keys:
            batch_data[k] = self.data[k]

        return batch_data

    @staticmethod
    def create_batch(items):
        """
        Create a batch from a list of :class:`_TensorDataManager` items.

        :param list[_TensorDataManager] items: A list of
            :class:`_TensorDataManager` items to batch.
        :return: A new instance of :class:`_BatchManager` containing the batched
            data.
        :rtype: _BatchManager
        """
        # Return None if no items are provided
        if not items:
            return None

        # Retrieve the first _TensorDataManager of the list
        first = items[0]

        # Initialize the batch manager
        batch_data = _BatchManager()

        # Iterate over the keys of the _TensorDataManager
        for k in first.keys:

            # Extract values and a sample used to determine the batch function
            vals = [it.data[k] for it in items]
            sample = vals[0]

            # Define the batch function based on the data type
            if isinstance(sample, (torch.Tensor, LabelTensor)):
                batch_fn = (
                    LabelTensor.stack
                    if isinstance(sample, LabelTensor)
                    else torch.stack
                )
                batch_data[k] = batch_fn(vals)

            # If no tensor is provided, just take the first value
            else:
                batch_data[k] = sample

        return batch_data
