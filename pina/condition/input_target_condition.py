"""
This module contains condition classes for supervised learning tasks.
"""

import torch
from copy import deepcopy
from torch_geometric.data import Data
from ..label_tensor import LabelTensor
from ..graph import Graph, LabelBatch
from .condition_base import ConditionBase
from torch_geometric.data import Batch


class InputTargetCondition(ConditionBase):
    """
    The :class:`InputTargetCondition` class represents a supervised condition
    defined by both ``input`` and ``target`` data. The model is trained to
    reproduce the ``target`` values given the ``input``. Supported data types
    include :class:`torch.Tensor`, :class:`~pina.label_tensor.LabelTensor`,
    :class:`~pina.graph.Graph`, or :class:`~torch_geometric.data.Data`.

    The class automatically selects the appropriate implementation based on
    the types of ``input`` and ``target``. Depending on whether the ``input``
    and ``target`` are tensors or graph-based data, one of the following
    specialized subclasses is instantiated:

    - :class:`TensorInputTensorTargetCondition`: For cases where both ``input``
      and ``target`` data are either :class:`torch.Tensor` or
      :class:`~pina.label_tensor.LabelTensor`.

    - :class:`TensorInputGraphTargetCondition`: For cases where ``input`` is
      either a :class:`torch.Tensor` or :class:`~pina.label_tensor.LabelTensor`
      and ``target`` is either a :class:`~pina.graph.Graph` or a
      :class:`torch_geometric.data.Data`.

    - :class:`GraphInputTensorTargetCondition`: For cases where ``input`` is
      either a :class:`~pina.graph.Graph` or :class:`torch_geometric.data.Data`
      and ``target`` is either a :class:`torch.Tensor` or a
      :class:`~pina.label_tensor.LabelTensor`.

    - :class:`GraphInputGraphTargetCondition`: For cases where both ``input``
      and ``target`` are either :class:`~pina.graph.Graph` or
      :class:`torch_geometric.data.Data`.

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
        Instantiate the appropriate subclass of :class:`InputTargetCondition`
        based on the types of both ``input`` and ``target`` data.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param target: The target data for the condition.
        :type target: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :return: The subclass of InputTargetCondition.
        :rtype: pina.condition.input_target_condition.
            TensorInputTensorTargetCondition |
            pina.condition.input_target_condition.
            TensorInputGraphTargetCondition |
            pina.condition.input_target_condition.
            GraphInputTensorTargetCondition |
            pina.condition.input_target_condition.GraphInputGraphTargetCondition

        :raises ValueError: If ``input`` and/or ``target`` are not of type
            :class:`torch.Tensor`, :class:`~pina.label_tensor.LabelTensor`,
            :class:`~pina.graph.Graph`, or :class:`~torch_geometric.data.Data`.
        """
        if cls != InputTargetCondition:
            return super().__new__(cls)

        # Tensor - Tensor
        if isinstance(input, (torch.Tensor, LabelTensor)) and isinstance(
            target, (torch.Tensor, LabelTensor)
        ):
            subclass = TensorInputTensorTargetCondition
            return subclass.__new__(subclass, input, target)

        # Tensor - Graph
        if isinstance(input, (torch.Tensor, LabelTensor)) and isinstance(
            target, (Graph, Data, list, tuple)
        ):
            cls._check_graph_list_consistency(target)
            subclass = TensorInputGraphTargetCondition
            return subclass.__new__(subclass, input, target)

        # Graph - Tensor
        if isinstance(input, (Graph, Data, list, tuple)) and isinstance(
            target, (torch.Tensor, LabelTensor)
        ):
            cls._check_graph_list_consistency(input)
            subclass = GraphInputTensorTargetCondition
            return subclass.__new__(subclass, input, target)

        raise ValueError(
            "Invalid input | target types."
            "Please provide either torch_geometric.data.Data, Graph, "
            "LabelTensor or torch.Tensor objects."
        )

    def __init__(self, input, target):
        """
        Initialization of the :class:`InputTargetCondition` class.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param target: The target data for the condition.
        :type target: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]

        .. note::

            If either ``input`` or ``target`` is a list of
            :class:`~pina.graph.Graph` or :class:`~torch_geometric.data.Data`
            objects, all elements in the list must share the same structure,
            with matching keys and consistent data types.
        """
        self._check_input_target_len(input, target)
        super().__init__(input=input, target=target)

    @staticmethod
    def _check_input_target_len(input, target):
        """
        Check that the length of the input and target lists are the same.

        :param input: The input data.
        :type input: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :param target: The target data.
        :type target: torch.Tensor | LabelTensor | Graph | Data | list[Graph] |
            list[Data] | tuple[Graph] | tuple[Data]
        :raises ValueError: If the lengths of the input and target lists do not
            match.
        """
        if isinstance(input, (Graph, Data)) or isinstance(
            target, (Graph, Data)
        ):
            return

        # Raise an error if the lengths of the input and target do not match
        if len(input) != len(target):
            raise ValueError(
                "The input and target lists must have the same length."
            )


class TensorInputTensorTargetCondition(InputTargetCondition):
    """
    Specialization of the :class:`InputTargetCondition` class for the case where
    both ``input`` and ``target`` are :class:`torch.Tensor` or
    :class:`~pina.label_tensor.LabelTensor` objects.
    """

    @property
    def input(self):
        """
        Return the input data for the condition.

        :return: The input data.
        :rtype: torch.Tensor | LabelTensor
        """
        return self.data["input"]

    @property
    def target(self):
        """
        Return the target data for the condition.

        :return: The target data.
        :rtype: torch.Tensor | LabelTensor
        """
        return self.data["target"]


class TensorInputGraphTargetCondition(InputTargetCondition):
    """
    Specialization of the :class:`InputTargetCondition` class for the case where
    ``input`` is either a :class:`torch.Tensor` or a
    :class:`~pina.label_tensor.LabelTensor` object and ``target`` is either a
    :class:`~pina.graph.Graph` or a :class:`torch_geometric.data.Data` object.
    """

    def _store_data(self, **kwargs):
        return self._store_graph_data(
            kwargs["target"], kwargs["input"], key="x"
        )

    @property
    def input(self):
        """
        Return the input data for the condition.

        :return: The input data.
        :rtype: list[torch.Tensor] | list[LabelTensor]
        """
        targets = []
        is_lt = isinstance(self.data["data"][0].x, LabelTensor)
        for graph in self.data["data"]:
            targets.append(graph.x)
        return torch.stack(targets) if not is_lt else LabelTensor.stack(targets)

    @property
    def target(self):
        """
        Return the target data for the condition.

        :return: The target data.
        :rtype: list[Graph] | list[Data]
        """
        return self.data["data"]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.get_multiple_data(idx)
        return {"data": self.data["data"][idx]}

    def get_multiple_data(self, indices):
        data = self.batch_fn([self.data["data"][i] for i in indices])
        x = data.x
        del data.x  # Avoid duplication of y on GPU memory
        return {
            "input": x,
            "target": data,
        }

    @classmethod
    def automatic_batching_collate_fn(cls, batch):
        """
        Collate function to be used in DataLoader.

        :param batch: A list of items from the dataset.
        :type batch: list
        :return: A collated batch.
        :rtype: dict
        """
        collated_graphs = super().automatic_batching_collate_fn(batch)
        x = collated_graphs["data"].x
        del collated_graphs["data"].x  # Avoid duplication of y on GPU memory
        to_return = {"input": x, "input": collated_graphs["data"]}
        return to_return


class GraphInputTensorTargetCondition(InputTargetCondition):
    """
    Specialization of the :class:`InputTargetCondition` class for the case where
    ``input`` is either a :class:`~pina.graph.Graph` or
    :class:`torch_geometric.data.Data` object and ``target`` is either a
    :class:`torch.Tensor` or a :class:`~pina.label_tensor.LabelTensor` object.
    """

    def __init__(self, input, target):
        """
        Initialization of the :class:`GraphInputTensorTargetCondition` class.

        :param input: The input data for the condition.
        :type input: Graph | Data | list[Graph] | list[Data] |
            tuple[Graph] | tuple[Data]
        :param target: The target data for the condition.
        :type target: torch.Tensor | LabelTensor
        """
        super().__init__(input=input, target=target)
        self.batch_fn = (
            LabelBatch.from_data_list
            if isinstance(input[0], Graph)
            else Batch.from_data_list
        )

    def _store_data(self, **kwargs):
        return self._store_graph_data(
            kwargs["input"], kwargs["target"], key="y"
        )

    @property
    def input(self):
        """
        Return the input data for the condition.

        :return: The input data.
        :rtype: list[Graph] | list[Data]
        """
        return self.data["data"]

    @property
    def target(self):
        """
        Return the target data for the condition.

        :return: The target data.
        :rtype: list[torch.Tensor] | list[LabelTensor]
        """
        targets = []
        is_lt = isinstance(self.data["data"][0].y, LabelTensor)
        for graph in self.data["data"]:
            targets.append(graph.y)

        return torch.stack(targets) if not is_lt else LabelTensor.stack(targets)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.get_multiple_data(idx)
        return {"data": self.data["data"][idx]}

    def get_multiple_data(self, indices):
        data = self.batch_fn([self.data["data"][i] for i in indices])
        y = data.y
        del data.y  # Avoid duplication of y on GPU memory
        return {
            "input": data,
            "target": y,
        }

    @classmethod
    def automatic_batching_collate_fn(cls, batch):
        """
        Collate function to be used in DataLoader.

        :param batch: A list of items from the dataset.
        :type batch: list
        :return: A collated batch.
        :rtype: dict
        """
        collated_graphs = super().automatic_batching_collate_fn(batch)
        y = collated_graphs["data"].y
        del collated_graphs["data"].y  # Avoid duplication of y on GPU memory
        print("y shape:", y.shape)
        print(y.labels)
        to_return = {"target": y, "input": collated_graphs["data"]}
        return to_return
