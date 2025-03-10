"""Module for PINA Scheduler."""

from abc import ABCMeta, abstractmethod


class Scheduler(metaclass=ABCMeta):
    """
    TODO

    :param metaclass: _description_, defaults to ABCMeta
    :type metaclass: _type_, optional
    """

    @property
    @abstractmethod
    def instance(self):
        """
        TODO
        """

    @abstractmethod
    def hook(self):
        """
        TODO
        """
