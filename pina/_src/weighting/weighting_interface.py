"""Module for the Weighting Interface."""

from abc import ABCMeta, abstractmethod


class WeightingInterface(metaclass=ABCMeta):
    """
    Abstract interface for all weighting schemas.
    """

    @abstractmethod
    def aggregate(self, losses):
        """
        Aggregate a collection of loss terms into a single scalar.

        This method applies the current weighting scheme to the provided losses
        and returns the aggregated result. Implementations may internally update
        the weights (e.g., based on training state) before performing the
        aggregation.

        :param dict losses: The mapping from loss names to loss tensors.
        :return: The aggregated loss value.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def update_weights(self, losses):
        """
        Update the weights based on the current losses.

        This method defines how the weighting strategy adapts over time. It is
        responsible for computing and storing updated weights that will be used
        during aggregation.

        :param dict losses: The mapping from loss names to loss tensors.
        :return: The updated weights.
        :rtype: dict
        """

    @abstractmethod
    def last_saved_weights(self):
        """
        Get the most recently computed weights.

        :return: The mapping from loss names to their corresponding weights.
        :rtype: dict
        """

    @property
    @abstractmethod
    def solver(self):
        """
        Solver associated with this weighting strategy.

        Provides access to the solver instance that uses this weighting scheme,
        enabling strategies that depend on training state or model information.

        :return: The solver instance.
        :rtype: :class:`~pina.solver.SolverInterface`
        """
