"""Module for Location class."""
import torch
from .location import Location
from .exclusion_domain import Exclusion
from ..utils import check_consistency
from ..label_tensor import LabelTensor
import random


class Intersection(Exclusion):
    """ PINA implementation of Intersection of Domains."""

    def __init__(self, geometries):
        """ PINA implementation of Intersection of Domains.

        :param list geometries: A list of geometries from 'pina.geometry' 
            such as 'EllipsoidDomain' or 'CartesianDomain'.

        :Example:
            # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})

            # Create a Intersection of the ellipsoid domains
            >>> intersection = Intersection([ellipsoid1, ellipsoid2])
        """
        super().__init__(geometries)

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

            # Create a Intersection of the ellipsoid domains
            >>> intersection = Intersection([cartesian1, cartesian2])

            >>> intersection.sample(n=1000)
                LabelTensor([[1.5562, 1.8656],
                            [1.1060, 1.2712],
                            [1.3909, 1.5579],
                            ...,
                            [1.5350, 1.7613],
                            [1.1835, 1.5107],
                            [1.1986, 1.6461]])

            >>> len(Intersection.sample(n=1000)
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
                if not self.is_inside(sample):
                    sampled_points.append(sample)
            sampled += sampled_points

        return LabelTensor(torch.cat(sampled), labels=[f'{i}' for i in self.variables])
