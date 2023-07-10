"""Module for Location class."""
import torch
from .exclusion_domain import Exclusion
from ..label_tensor import LabelTensor
from .operation_interface import OperationInterface
import random


class Intersection(OperationInterface):
    """ PINA implementation of Intersection of Domains."""

    def __init__(self, geometries):
        """ PINA implementation of Intersection of Domains.

        :param list geometries: A list of geometries from 'pina.geometry' 
            such as 'EllipsoidDomain' or 'CartesianDomain'. The intersection
            will be taken between all the geometries in the list. The resulting
            geometry will be the intersection of all the geometries in the list.

        :Example:
            # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})

            # Create a Intersection of the ellipsoid domains
            >>> intersection = Intersection([ellipsoid1, ellipsoid2])
        """
        super().__init__(geometries)

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
        return flag == len(self.geometries)

    def sample(self, n, mode='random', variables='all'):
        """Sample routine for intersection domain.

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

            >>> intersection.sample(n=5)
                LabelTensor([[1.7697, 1.8654],
                            [1.2841, 1.1208],
                            [1.7289, 1.9843],
                            [1.3332, 1.2448],
                            [1.9902, 1.4458]])

            >>> len(intersection.sample(n=5)
                5

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
                if self.is_inside(sample):
                    sampled_points.append(sample)
            sampled += sampled_points

        return LabelTensor(torch.cat(sampled), labels=self.variables)
