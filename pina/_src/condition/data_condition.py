"""Module for the Data Condition class."""

import torch
from torch_geometric.data import Data
from pina._src.condition.base_condition import BaseCondition
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.graph import Graph
from pina._src.condition.data_manager import _DataManager
from pina._src.core.utils import check_consistency


class DataCondition(BaseCondition):
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

    # Available fields, input and conditional variables data types
    __fields__ = ["input", "conditional_variables"]
    _avail_input_cls = (torch.Tensor, LabelTensor, Data, Graph)
    _avail_conditional_variables_cls = (torch.Tensor, LabelTensor)

    def __new__(cls, input, conditional_variables=None):
        """
        Check the types of ``input`` and ``conditional_variables`` and
        instantiate an instance of :class:`DataCondition` accordingly.

        :param input: The input data associated with the condition.
        :type input: torch.Tensor | LabelTensor | Graph |
            Data | list[Graph] | list[Data] | tuple[Graph] | tuple[Data]
        :param conditional_variables: The conditional variables associated with
            the condition. Default is ``None``.
        :type conditional_variables: torch.Tensor | LabelTensor
        :raises ValueError: If ``input`` is not of type :class:`torch.Tensor`,
            :class:`~pina.label_tensor.LabelTensor`, :class:`~pina.graph.Graph`,
            or :class:`~torch_geometric.data.Data`, nor is it a list or tuple of
            :class:`~pina.graph.Graph` or :class:`~torch_geometric.data.Data`.
        :raises ValueError: If ``conditional_variables`` is not of type
            :class:`torch.Tensor` or :class:`~pina.label_tensor.LabelTensor`.
        :return: A new instance of :class:`DataCondition`.
        :rtype: DataCondition
        """
        # Check input type - if iterable, ensure it is either Data or Graph
        if isinstance(input, (list, tuple)):
            check_consistency(input, (Data, Graph))
        else:
            check_consistency(input, cls._avail_input_cls)

        # Check conditional_variables type
        if conditional_variables is not None:
            check_consistency(
                conditional_variables, cls._avail_conditional_variables_cls
            )

        return super().__new__(cls)

    def store_data(self, **kwargs):
        """
        Store the input data and the conditional variables in a dictionary-like
        structure.

        :param dict kwargs: The keyword arguments containing the data to be
            stored.
        :return: A dictionary-like structure containing the stored data.
        :rtype: _DataManager
        """
        # Store input and conditional variables in a dictionary-like structure
        data_dict = {"input": kwargs.get("input")}
        cond_vars = kwargs.get("conditional_variables", None)
        if cond_vars is not None:
            data_dict["conditional_variables"] = cond_vars

        return _DataManager(**data_dict)

    @property
    def conditional_variables(self):
        """
        The conditional variables associated with the condition.

        :return: The conditional variables.
        :rtype: torch.Tensor | LabelTensor | None
        """
        if hasattr(self.data, "conditional_variables"):
            return self.data.conditional_variables

        return None

    @property
    def input(self):
        """
        The input data associated with the condition.

        :return: The input data.
        :rtype: torch.Tensor | LabelTensor | Graph | Data |
            list[Graph] | list[Data] | tuple[Graph] | tuple[Data]
        """
        return self.data.input
