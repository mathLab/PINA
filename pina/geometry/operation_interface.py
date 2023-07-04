import torch
from .location import Location
from ..utils import check_consistency
from ..label_tensor import LabelTensor
from abc import ABCMeta, abstractmethod
import random


class OperationInterface(Location, metaclass=ABCMeta):
    def __init__(self, geometries):
        """
        Abstract Operation class.
        Any geometry operation entity must inherit from this class.

        :param list geometries: A list of geometries from 'pina.geometry' 
            such as 'EllipsoidDomain' or 'CartesianDomain'.
        """
        # check consistency geometries
        check_consistency(geometries, Location)

        # check we are passing always different 
        # geometries with the same labels.
        self._check_dimensions(geometries)
        
        # assign geometries
        self._geometries = geometries

    @property
    def geometries(self):
        """ 
        The geometries."""
        return self._geometries

    @property
    def variables(self):
        """
        Spatial variables.

        :return: All the variables defined in ``__init__`` in order.
        :rtype: list[str]
        """
        return self.geometries[0].variables

    def _check_dimensions(self, geometries):
        """Check if the dimensions of the geometries are consistent.

        :param geometries: Geometries to be checked.
        :type geometries: list[Location]
        """
        for geometry in geometries:
            if geometry.variables != geometries[0].variables:
                raise NotImplementedError(
                    f'The geometries need to have same dimensions and labels.')
