"""Module for the Condition interface."""

from abc import ABCMeta, abstractmethod


class ConditionInterface(metaclass=ABCMeta):
    """
    Abstract base class for PINA conditions. All specific conditions must
    inherit from this interface.

    Refer to :class:`pina.condition.condition.Condition` for a thorough
    description of all available conditions and how to instantiate them.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialization of the :class:`ConditionInterface` class.
        """

    @property
    @abstractmethod
    def problem(self):
        """
        Return the problem associated with this condition.

        :return: Problem associated with this condition.
        :rtype: ~pina.problem.abstract_problem.AbstractProblem
        """

    @problem.setter
    @abstractmethod
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param pina.problem.abstract_problem.AbstractProblem value: The problem
            to associate with this condition
        """

    @abstractmethod
    def __len__(self):
        """
        Return the number of data points in the condition.

        :return: Number of data points.
        :rtype: int
        """

    @abstractmethod
    def __getitem__(self, idx):
        """
        Return the data point(s) at the specified index.

        :param int idx: Index of the data point(s) to retrieve.
        :return: Data point(s) at the specified index.
        """
