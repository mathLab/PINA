"""Module for PINA Optimizer."""

from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):
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
