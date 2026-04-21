"""Module for the Data Manager factory class."""

import torch
from pina._src.core.label_tensor import LabelTensor
from pina._src.equation.base_equation import BaseEquation
from pina._src.condition.graph_data_manager import _GraphDataManager
from pina._src.condition.tensor_data_manager import _TensorDataManager


class _DataManager:
    """
    Factory class for data manager implementations.

    This class dispatches object creation to either
    :class:`~pina.condition.tensor_data_manager._TensorDataManager` or
    :class:`~pina.condition.graph_data_manager._GraphDataManager` depending on
    the types of the provided keyword arguments.
    """

    def __new__(cls, **kwargs):
        """
        Create the appropriate data manager implementation based on the provided
        keyword arguments.

        If all values in ``kwargs`` are instances of :class:`torch.Tensor`,
        :class:`~pina.label_tensor.LabelTensor`, or
        :class:`~pina.equation.base_equation.BaseEquation`, an instance of
        :class:`~pina.condition.tensor_data_manager._TensorDataManager` is
        created. Otherwise, an instance of
        :class:`~pina.condition.graph_data_manager._GraphDataManager` is
        created.

        :param dict kwargs: The keyword arguments for the data manager.
        :return: A concrete data manager instance.
        :rtype: _TensorDataManager | _GraphDataManager
        """
        # Guard subclass instantiation
        if cls is not _DataManager:
            return super().__new__(cls)

        # Check if there are only tensors / equations
        is_tensor_only = all(
            isinstance(v, (torch.Tensor, LabelTensor, BaseEquation))
            for v in kwargs.values()
        )

        # Choose the appropriate subclass
        subclass = _TensorDataManager if is_tensor_only else _GraphDataManager

        return subclass(**kwargs)
