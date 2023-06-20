"""Module for Location class."""

from abc import ABCMeta, abstractmethod


class Location(metaclass=ABCMeta):
    """
    Abstract Location class.
    Any geometry entity should inherit from this class.
    """
    @abstractmethod
    def sample(self):
        """
        Abstract method for sampling a point from the location. To be
        implemented in the child class.
        """
        pass

    @abstractmethod
    def is_inside(self):
        """
        Abstract method for checking if a point is inside the location. To be
        implemented in the child class.
        """
        pass