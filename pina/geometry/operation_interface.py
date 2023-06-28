import torch
from .location import Location
from ..utils import check_consistency
from ..label_tensor import LabelTensor
from abc import ABCMeta, abstractmethod
import random


class OperationInterface(Location, metaclass=ABCMeta):
    def __init__(self, geometries):

        # Exclusion checks
        check_consistency(geometries, Location)
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

        :return: All the spatial variables defined in '__init__()' in order.
        :rtype: list[str]
        """
        all_variables = []
        seen_variables = set()
        for geometry in self.geometries:
            for variable in geometry.variables:
                if variable not in seen_variables:
                    all_variables.append(variable)
                    seen_variables.add(variable)
        return all_variables

    def sample(self, n, type, mode='random', variables='all', ):
        """Sample points from the Exclusion domain.

        :param int n: Number of points to sample.
        :param str mode: Mode for sampling, defaults to 'random'.
            Available modes include: random sampling, 'random'.
        :param str variables: pinn variable to be sampled, defaults to 'all'.
        :param str type: Type of operation. Can be Union, Intersection, Exclusion, or Difference.

        """

        # check if mode is valid
        if mode != 'random':
            raise NotImplementedError(
                f'{mode} is not a valid mode for sampling.')

        sampled_points = []

        # calculate the number of points to sample for each geometry and the remainder.
        remainder = n % len(self.geometries)
        num_points = n // len(self.geometries)

        # sample points from each geometry

        # sample difference
        if type == 'difference':
            while len(sampled_points) < n:
                # get sample point from first geometry
                point = self.first.sample(1, mode, variables)
                if self.is_inside(point):
                    sampled_points.append(point)
        else:
            # randomize order to sample union, intersection, and exclusion
            iter_ = random.sample(self.geometries, len(self.geometries))
            for i, geometry in enumerate(iter_):
                if type == 'union':
                    # sample all the points from the geometry. They will all be in the shape.
                    sampled_points.append(geometry.sample(
                        num_points + int(i < remainder), mode, variables))
                    # in case number of sampled points is smaller than the number of geometries
                    if len(sampled_points) >= n:
                        break
                else:
                    sampled = []
                    while len(sampled) < (num_points + int(i < remainder)):
                        # sample points from the geometry until the number of points is reached.
                        sample = geometry.sample(1, mode, variables)
                        if self.is_inside(sample):
                            sampled.append(sample)
                    sampled_points += sampled

        return LabelTensor(torch.cat(sampled_points), labels=[f'{i}' for i in self.variables])

    def _check_dimensions(self, geometries):
        """Check if the dimensions of the geometries are consistent.

        :param geometries: Geometries to be checked.
        :type geometries: list[Location]
        """
        for geometry in geometries:
            if geometry.variables != geometries[0].variables:
                raise NotImplementedError(
                    f'The geometries need to be the same dimensions. {geometry.variables} is not equal to {geometries[0].variables}')
