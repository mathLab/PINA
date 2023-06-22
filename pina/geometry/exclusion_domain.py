"""Module for Location class."""
import torch
from .location import Location
from ..utils import check_consistency
from ..label_tensor import LabelTensor
import random


class Exclusion(Location):
    """ PINA implementation of Exclusion of Domains."""

    def __init__(self, geometries):
        """ PINA implementation of Exclusion of Domains.

        :param list geometries: A list of geometries from 'pina.geometry' 
            such as 'EllipsoidDomain' or 'CartesianDomain'.

        :Example:
            # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})

            # Create a Exclusion of the ellipsoid domains
            >>> exclusion = Exclusion([ellipsoid1, ellipsoid2])
        """
        super().__init__()

        # Exclusion checks
        check_consistency(geometries, Location)
        self._check_exclusion_dimesions(geometries)

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

    def is_inside(self, point, check_border=False):
        """Check if a point is inside the Exclusion domain.

        :param point: Point to be checked.
        :type point: torch.Tensor   
        :param bool check_border: If True, the border is considered inside.
        :return: True if the point is inside the Exclusion domain, False otherwise.
        :rtype: bool
        """
        flag = 0
        for geometry in self.geometries:
            if geometry.is_inside(point, check_border):
                flag += 1
        return flag == 1

    def sample(self, n, mode='random', variables='all'):
        """Sample routine.

        :param n: Number of points to sample in the shape.
        :type n: int
        :param mode: Mode for sampling, defaults to 'random'.
            Available modes include: random sampling, 'random'.
        :type mode: str, optional
        :param variables: pinn variable to be sampled, defaults to 'all'.
        :type variables: str or list[str], optional

        :Example:
            # Create two Cartesian domains
            >>> cartesian1 = CartesianDomain({'x': [0, 2], 'y': [0, 2]})
            >>> cartesian2 = CartesianDomain({'x': [1, 3], 'y': [1, 3]})

            # Create a Exclusion of the ellipsoid domains
            >>> Exclusion = Exclusion([cartesian1, cartesian2])

            >>> Exclusion.sample(n=1000)
                LabelTensor([[1.6802, 0.7723],
                    [1.8872, 0.9972],
                    [0.3166, 0.1213],
                    ...,
                    [2.1999, 2.0767],
                    [1.0655, 2.7711],
                    [1.6751, 2.3695]])

            >>> len(Exclusion.sample(n=1000)
                1000

        """
        if mode != 'random':
            raise NotImplementedError(
                f'{mode} is not a valid mode for sampling.')

        sampled = []

        # calculate the number of points to sample for each geometry and the remainder.
        remainder = n % len(self.geometries)
        num_points = n // len(self.geometries)

        # sample the points
        # NB. geometries as shuffled since if we sample
        # multiple times just one point, we would end
        # up sampling only from the first geometry.
        iter_ = random.sample(self.geometries, len(self.geometries))
        for i, geometry in enumerate(iter_):
            sampled_points = []
            # int(i < remainder) is one only if we have a remainder
            # different than zero. Notice that len(geometries) is
            # always smaller than remaider.
            # makes sure point is uniquely inside 1 shape.
            while len(sampled_points) < (num_points + int(i < remainder)):
                sample = geometry.sample(1, mode, variables)
                # if not self.is_inside(sample) --> will be the intersection
                if self.is_inside(sample):
                    sampled_points.append(sample)
            sampled += sampled_points

        return LabelTensor(torch.cat(sampled), labels=[f'{i}' for i in self.variables])

    def _check_exclusion_dimesions(self, geometries):
        """Check if the dimensions of the geometries are consistent.

        :param geometries: Geometries to be checked.
        :type geometries: list[Location]
        """
        for geometry in geometries:
            if geometry.variables != geometries[0].variables:
                raise NotImplementedError(
                    f'The geometries need to be the same dimensions. {geometry.variables}'
                    f'is not equal to {geometries[0].variables}.')