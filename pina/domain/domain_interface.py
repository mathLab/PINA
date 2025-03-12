"""Module for the Domain Interface."""

from abc import ABCMeta, abstractmethod


class DomainInterface(metaclass=ABCMeta):
    """
    Abstract base class for geometric domains. All specific domain types should
    inherit from this class.
    """

    available_sampling_modes = ["random", "grid", "lh", "chebyshev", "latin"]

    @property
    @abstractmethod
    def sample_modes(self):
        """
        Abstract method defining sampling methods.
        """

    @property
    @abstractmethod
    def variables(self):
        """
        Abstract method returning the domain variables.
        """

    @sample_modes.setter
    def sample_modes(self, values):
        """
        Setter for the sample_modes property.

        :param values: Sampling modes to be set.
        :type values: str | list[str]
        :raises TypeError: Invalid sampling mode.
        """
        if not isinstance(values, (list, tuple)):
            values = [values]
        for value in values:
            if value not in DomainInterface.available_sampling_modes:
                raise TypeError(
                    f"mode {value} not valid. Expected at least "
                    "one in "
                    f"{DomainInterface.available_sampling_modes}."
                )

    @abstractmethod
    def sample(self):
        """
        Abstract method for the sampling routine.
        """

    @abstractmethod
    def is_inside(self, point, check_border=False):
        """
        Abstract method for checking if a point is inside the domain.

        :param LabelTensor point: Point to be checked.
        :param bool check_border: If ``True``, the border is considered inside
            the domain. Default is ``False``.
        """
