"""Module for the DomainInterface class."""

from abc import ABCMeta, abstractmethod


class DomainInterface(metaclass=ABCMeta):
    """
    Abstract Location class.
    Any geometry entity should inherit from this class.
    """

    available_sampling_modes = ["random", "grid", "lh", "chebyshev", "latin"]

    @property
    @abstractmethod
    def sample_modes(self):
        """
        Abstract method returing available samples modes for the Domain.
        """
        pass

    @property
    @abstractmethod
    def variables(self):
        """
        Abstract method returing Domain variables.
        """
        pass

    @sample_modes.setter
    def sample_modes(self, values):
        """
        TODO
        """
        if not isinstance(values, (list, tuple)):
            values = [values]
        for value in values:
            if value not in DomainInterface.available_sampling_modes:
                raise TypeError(f"mode {value} not valid. Expected at least "
                                "one in "
                                f"{DomainInterface.available_sampling_modes}."
                                )

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
