"""
Module that defines the ConditionInterface class.
"""

from abc import ABCMeta
from torch_geometric.data import Data
from ..label_tensor import LabelTensor
from ..graph import Graph


class ConditionInterface(metaclass=ABCMeta):
    """
    Abstract class which defines a common interface for all the conditions.
    It defined a common interface for all the conditions.

    """

    def __init__(self):
        """
        Initialize the ConditionInterface object.
        """

        self._problem = None

    @property
    def problem(self):
        """
        Return the problem to which the condition is associated.

        :return: Problem to which the condition is associated
        :rtype: pina.problem.AbstractProblem
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        """
        Set the problem to which the condition is associated.

        :param pina.problem.abstract_problem.AbstractProblem value: Problem to
            which the condition is associated
        """
        self._problem = value

    @staticmethod
    def _check_graph_list_consistency(data_list):
        """
        Check the consistency of the list of Data/Graph objects. It performs
        the following checks:

        1. All elements in the list must be of the same type (either Data or
        Graph).
        2. All elements in the list must have the same keys.
        3. The type of each tensor must be consistent across all elements in
        the list.
        4. If the tensor is a LabelTensor, the labels must be consistent across
        all elements in the list.

        :param data_list: List of Data/Graph objects to check
        :type data_list: list[Data] | list[Graph] | tuple[Data] | tuple[Graph]

        :raises ValueError:  If the input types are invalid.
        :raises ValueError: If all elements in the list do not have the same
            keys.
        :raises ValueError: If the type of each tensor is not consistent across
            all elements in the list.
        :raises ValueError: If the labels of the LabelTensors are not consistent
            across all elements in the list.
        """

        # If the data is a Graph or Data object, return (do not need to check
        # anything)
        if isinstance(data_list, (Graph, Data)):
            return

        # check all elements in the list are of the same type
        if not all(isinstance(i, (Graph, Data)) for i in data_list):
            raise ValueError(
                "Invalid input types. "
                "Please provide either Data or Graph objects."
            )
        data = data_list[0]
        # Store the keys of the first element in the list
        keys = sorted(list(data.keys()))

        # Store the type of each tensor inside first element Data/Graph object
        data_types = {name: tensor.__class__ for name, tensor in data.items()}

        # Store the labels of each LabelTensor inside first element Data/Graph
        # object
        labels = {
            name: tensor.labels
            for name, tensor in data.items()
            if isinstance(tensor, LabelTensor)
        }

        # Iterate over the list of Data/Graph objects
        for data in data_list[1:]:
            # Check if the keys of the current element are the same as the first
            # element
            if sorted(list(data.keys())) != keys:
                raise ValueError(
                    "All elements in the list must have the same keys."
                )
            for name, tensor in data.items():
                # Check if the type of each tensor inside the current element
                # is the same as the first element
                if tensor.__class__ is not data_types[name]:
                    raise ValueError(
                        f"Data {name} must be a {data_types[name]}, got "
                        f"{tensor.__class__}"
                    )
                # If the tensor is a LabelTensor, check if the labels are the
                # same as the first element
                if isinstance(tensor, LabelTensor):
                    if tensor.labels != labels[name]:
                        raise ValueError(
                            "LabelTensor must have the same labels"
                        )
