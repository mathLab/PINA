"""Module for the Operation Interface."""

from abc import ABCMeta, abstractmethod
from .domain_interface import DomainInterface
from ..utils import check_consistency


class OperationInterface(DomainInterface, metaclass=ABCMeta):
    """
    Abstract class for set operations defined on geometric domains.
    """

    def __init__(self, geometries):
        """
        Initialization of the :class:`OperationInterface` class.

        :param list[DomainInterface] geometries: A list of instances of the
            :class:`~pina.domain.domain_interface.DomainInterface` class on
            which the set operation is performed.
        """
        # check consistency geometries
        check_consistency(geometries, DomainInterface)

        # check we are passing always different
        # geometries with the same labels.
        self._check_dimensions(geometries)

        # assign geometries
        self._geometries = geometries

    @property
    def sample_modes(self):
        """
        List of available sampling modes.

        :return: List of available sampling modes.
        :rtype: list[str]
        """
        return ["random"]

    @property
    def geometries(self):
        """
        The domains on which to perform the set operation.

        :return: The domains on which to perform the set operation.
        :rtype: list[DomainInterface]
        """
        return self._geometries

    @property
    def variables(self):
        """
        List of variables of the domain.

        :return: List of variables of the domain.
        :rtype: list[str]
        """
        variables = []
        for geom in self.geometries:
            variables += geom.variables
        return sorted(list(set(variables)))

    @abstractmethod
    def is_inside(self, point, check_border=False):
        """
        Abstract method to check if a point lies inside the resulting domain
        after performing the set operation.

        :param LabelTensor point: Point to be checked.
        :param bool check_border: If ``True``, the border is considered inside
            the resulting domain. Default is ``False``.
        :return: ``True`` if the point is inside the domain,
            ``False`` otherwise.
        :rtype: bool
        """

    def _check_dimensions(self, geometries):
        """
        Check if the dimensions of the geometries are consistent.

        :param list[DomainInterface] geometries: Domains to be checked.
        :raises NotImplementedError: If the dimensions of the geometries are not
            consistent.
        """
        for geometry in geometries:
            if geometry.variables != geometries[0].variables:
                raise NotImplementedError(
                    "The geometries need to have same dimensions and labels."
                )
