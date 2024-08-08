"""Module for the DomainInterface class."""

from abc import ABCMeta, abstractmethod


class DomainInterface(metaclass=ABCMeta):
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
    def is_inside(self, point, check_border=False):
        """
        Abstract method for checking if a point is inside the location. To be
        implemented in the child class.

        :param torch.Tensor point: A tensor point to be checked.
        :param bool check_border: A boolean that determines whether the border
            of the location is considered checked to be considered inside or
            not. Defaults to ``False``.
        """
        pass
