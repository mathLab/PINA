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
    """

    def __init__(self):
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
        self._problem = value

    @staticmethod
    def _check_graph_list_consistency(data_list):

        # If the data is a Graph or Data object, return (do not need to check
        # anything)
        if isinstance(data_list, (Graph, Data)):
            return
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
        print(data_types)
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
