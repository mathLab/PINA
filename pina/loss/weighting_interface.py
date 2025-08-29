"""Module for the Weighting Interface."""

from abc import ABCMeta, abstractmethod


class WeightingInterface(metaclass=ABCMeta):
    """
    Abstract base class for all loss weighting schemas. All weighting schemas
    should inherit from this class.
    """

    def __init__(self):
        """
        Initialization of the :class:`WeightingInterface` class.
        """
        self._solver = None

    @abstractmethod
    def aggregate(self, losses):
        """
        Aggregate the losses.

        :param dict losses: The dictionary of losses.
        """

    @property
    def solver(self):
        """
        The solver employing this weighting schema.

        :return: The solver.
        :rtype: :class:`~pina.solver.SolverInterface`
        """
        return self._solver
