"""Module for the Scheduler Interface."""

from abc import ABCMeta, abstractmethod


class SchedulerInterface(metaclass=ABCMeta):
    """
    Abstract interface for all schedulers.
    """

    @abstractmethod
    def hook(self, optimizer):
        """
        Execute custom logic associated with the scheduler instance.

        This method is intended to encapsulate any additional behavior that
        should be triggered during the optimization process.

        :param OptimizerInterface optimizer: The optimizer instance associated
            with the scheduler.
        """

    @property
    @abstractmethod
    def instance(self):
        """
        The underlying scheduler object.

        :return: The scheduler instance.
        :rtype: object
        """
