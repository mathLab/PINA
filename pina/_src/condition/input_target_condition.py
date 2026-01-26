"""
This module contains condition classes for supervised learning tasks.
"""

import torch
from torch_geometric.data import Data
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.graph import Graph
from pina._src.condition.condition_base import ConditionBase
from pina._src.condition.data_manager import _DataManager


class InputTargetCondition(ConditionBase):
    """
    The :class:`InputTargetCondition` class represents a supervised condition
    defined by both ``input`` and ``target`` data. The model is trained to
    reproduce the ``target`` values given the ``input``. Supported data types
    include :class:`torch.Tensor`, :class:`~pina.label_tensor.LabelTensor`,
    :class:`~pina.graph.Graph`, or :class:`~torch_geometric.data.Data`.

    :Example:

    >>> from pina import Condition, LabelTensor
    >>> from pina.graph import Graph
    >>> import torch

    >>> pos = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> edge_index = torch.randint(0, 100, (2, 300))
    >>> graph = Graph(pos=pos, edge_index=edge_index)

    >>> input = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> condition = Condition(input=input, target=graph)
    """

    # Available input and target data types
    __fields__ = ["input", "target"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)
    _avail_output_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)

    def __new__(cls, input, target):
        """
        Check the types of ``input`` and ``target`` data and instantiate the
        :class:`InputTargetCondition`.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph |
            Data | list[Graph] | list[Data] | tuple[Graph] | tuple[Data]
        :param target: The target data for the condition.
        :type target: torch.Tensor | LabelTensor | Graph |
            Data | list[Graph] | list[Data] | tuple[Graph] | tuple[Data]
        :return: An instance of :class:`InputTargetCondition`.
        :rtype: pina.condition.input_target_condition.InputTargetCondition
        :raises ValueError: If ``input`` or ``target`` are not of supported types.
        """

        if not isinstance(input, cls._avail_input_cls):
            raise ValueError(
                "Invalid input type. Expected one of the following: "
                "torch.Tensor, LabelTensor, Graph, Data or "
                "list/tuple of Graph/Data objects."
            )
        if isinstance(input, (list, tuple)):
            for item in input:
                if not isinstance(item, (Graph, Data)):
                    raise ValueError(
                        "If target is a list or tuple, all its elements "
                        "must be of type Graph or Data."
                    )

        if not isinstance(target, cls._avail_output_cls):
            raise ValueError(
                "Invalid target type. Expected one of the following: "
                "torch.Tensor, LabelTensor, Graph, Data or "
                "list/tuple of Graph/Data objects."
            )
        if isinstance(target, (list, tuple)):
            for item in target:
                if not isinstance(item, (Graph, Data)):
                    raise ValueError(
                        "If target is a list or tuple, all its elements "
                        "must be of type Graph or Data."
                    )

        return super().__new__(cls)

    def store_data(self, **kwargs):
        """
        Store the input and target data in a :class:`_DataManager` object.
        :param dict kwargs: The keyword arguments containing the input and
            target data.
        """
        return _DataManager(**kwargs)

    @property
    def input(self):
        """
        Return the input data for the condition.

        :return: The input data.
        :rtype: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        """
        return self.data.input

    @property
    def target(self):
        """
        Return the target data for the condition.

        :return: The target data.
        :rtype: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        """
        return self.data.target
