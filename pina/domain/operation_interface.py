""" Module for OperationInterface class. """

from .domain_interface import DomainInterface
from ..utils import check_consistency
from abc import ABCMeta, abstractmethod


class OperationInterface(DomainInterface, metaclass=ABCMeta):

    def __init__(self, geometries):
        """
        Abstract set operation class. Any geometry operation entity must inherit from this class.

        :param list geometries: A list of geometries from ``pina.geometry``
            such as ``EllipsoidDomain`` or ``CartesianDomain``.
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
        return ["random"]

    @property
    def geometries(self):
        """
        The geometries to perform set operation.
        """
        return self._geometries

    @property
    def variables(self):
        """
        Spatial variables of the domain.

        :return: All the variables defined in ``__init__`` in order.
        :rtype: list[str]
        """
        variables = []
        for geom in self.geometries:
            variables += geom.variables
        return sorted(list(set(variables)))

    @abstractmethod
    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the resulting domain after
        a set operation is applied.

        :param point: Point to be checked.
        :type point: torch.Tensor
        :param bool check_border: If ``True``, the border is considered inside.
        :return: ``True`` if the point is inside the Intersection domain, ``False`` otherwise.
        :rtype: bool
        """
        pass

    def _check_dimensions(self, geometries):
        """Check if the dimensions of the geometries are consistent.

        :param geometries: Geometries to be checked.
        :type geometries: list[Location]
        """
        for geometry in geometries:
            if geometry.variables != geometries[0].variables:
                raise NotImplementedError(
                    f"The geometries need to have same dimensions and labels."
                )
