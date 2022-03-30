"""Module for Location class."""

from abc import ABCMeta, abstractmethod


class Location(metaclass=ABCMeta):
    """
    Abstract class
    """

    @property
    @abstractmethod
    def sample(self):
        pass
