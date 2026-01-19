"""Module for the DataCondition class."""

import torch
from torch_geometric.data import Data
from .condition_base import ConditionBase
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..condition.data_manager import _DataManager


class DataCondition(ConditionBase):
    """
    The class :class:`DataCondition` defines an unsupervised condition based on
    ``input`` data. This condition is typically used in data-driven problems,
    where the model is trained using a custom unsupervised loss determined by
    the chosen :class:`~pina.solver.solver.SolverInterface`, while leveraging
    the provided data during training. Optional ``conditional_variables`` can be
    specified when the model depends on additional parameters.

    :Example:

    >>> from pina import Condition, LabelTensor
    >>> import torch

    >>> pts = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> cond_vars = LabelTensor(torch.randn(100, 1), labels=["w"])
    >>> condition = Condition(input=pts, conditional_variables=cond_vars)
    """

    # Available input data types
    __fields__ = ["input", "conditional_variables"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph, list, tuple)
    _avail_conditional_variables_cls = (torch.Tensor, LabelTensor)

    def __new__(cls, input, conditional_variables=None):
        """
        Check the types of ``input`` and ``conditional_variables`` and
        instantiate a class of :class:`DataCondition` accordingly.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph |
            Data | list[Graph] | list[Data] | tuple[Graph] | tuple[Data]
        :param conditional_variables: The conditional variables for the
            condition. Default is ``None``.
        :type conditional_variables: torch.Tensor | LabelTensor
        :return: The subclass of DataCondition.
        :rtype: pina.condition.data_condition.TensorDataCondition |
            pina.condition.data_condition.GraphDataCondition
        :raises ValueError: If ``input`` is not of type :class:`torch.Tensor`,
            :class:`~pina.label_tensor.LabelTensor`, :class:`~pina.graph.Graph`,
            or :class:`~torch_geometric.data.Data`.
        """
        if cls != DataCondition:
            return super().__new__(cls)

        # Check input type
        if not isinstance(input, cls._avail_input_cls):
            raise ValueError(
                "Invalid input type. Expected one of the following: "
                "torch.Tensor, LabelTensor, Graph, Data or "
                "an iterable of the previous types."
            )
        if isinstance(input, (list, tuple)):
            for item in input:
                if not isinstance(item, (Data, Graph)):
                    raise ValueError(
                        "if input is a list or tuple, all its elements must"
                        " be of type Graph or Data."
                    )

        # Check conditional_variables type
        if conditional_variables is not None:
            if not isinstance(
                conditional_variables, cls._avail_conditional_variables_cls
            ):
                raise ValueError(
                    "Invalid conditional_variables type. Expected one of the "
                    "following: torch.Tensor, LabelTensor."
                )

        return super().__new__(cls)

    def store_data(self, **kwargs):
        """
        Store the input data and conditional variables in a dictionary.

        :param input: The input data for the condition.
        :type input: torch.Tensor | LabelTensor | Graph |
            Data | list[Graph] | list[Data] | tuple[Graph] | tuple[Data]
        :param conditional_variables: The conditional variables for the
            condition.
        :type conditional_variables: torch.Tensor | LabelTensor
        :return: A dictionary containing the stored data.
        :rtype: dict
        """
        data_dict = {"input": kwargs.get("input")}
        cond_vars = kwargs.get("conditional_variables", None)
        if cond_vars is not None:
            data_dict["conditional_variables"] = cond_vars
        return _DataManager(**data_dict)

    @property
    def conditional_variables(self):
        """
        Return the conditional variables for the condition.

        :return: The conditional variables.
        :rtype: torch.Tensor | LabelTensor | None
        """
        if hasattr(self.data, "conditional_variables"):
            return self.data.conditional_variables
        return None

    @property
    def input(self):
        """
        Return the input data for the condition.

        :return: The input data.
        :rtype: torch.Tensor | LabelTensor | Graph | Data |
            list[Graph] | list[Data] | tuple[Graph] | tuple[Data]
        """
        return self.data.input
