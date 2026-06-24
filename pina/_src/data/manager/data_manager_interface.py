"""Module for the Data Manager interface."""

from abc import ABCMeta, abstractmethod


class _DataManagerInterface(metaclass=ABCMeta):
    """
    Abstract interface for all data managers.

    :Example:

        >>> import torch
        >>> class CustomManager(_DataManagerInterface):
        ...     def __init__(self, data): self.data = data
        ...     def __len__(self): return len(self.data)
        ...     def __getitem__(self, idx): return self.data[idx]
        ...     def to_batch(self): return self
        ...     @staticmethod
        ...     def create_batch(items): return items
        >>> manager = CustomManager([1, 2, 3])
        >>> len(manager)
        3
    """

    @abstractmethod
    def __len__(self):
        """
        Return the number of samples in the data manager.

        :return: The number of samples.
        :rtype: int
        """

    @abstractmethod
    def __getitem__(self, idx):
        """
        Return the item at the specified indices.

        :param idx: The indices of the data point to retrieve.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: A new :class:`_DataManager` instance containing the
            selected data items.
        :rtype: _DataManager
        """

    @abstractmethod
    def to_batch(self):
        """
        Create a batch from the current data manager.

        :return: A new :class:`~pina.condition.data_manager._DataManager`
            instance with batched data.
        :rtype: _DataManager
        """

    @staticmethod
    @abstractmethod
    def create_batch(items):
        """
        Create a batch from a list of :class:`_DataManager` items.

        :param list[_DataManager] items: A list of
            :class:`_DataManager` items to batch.
        :return: A new instance of :class:`_DataManager` containing the
            batched data.
        :rtype: _DataManager
        """
