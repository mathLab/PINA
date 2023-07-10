"""Module for Location class."""
import torch
from .exclusion_domain import Exclusion
from .operation_interface import OperationInterface
from ..label_tensor import LabelTensor


class Difference(OperationInterface):
    """ PINA implementation of Difference of Domains."""

    def __init__(self, geometries):
        """ PINA implementation of Difference of Domains.

        :param list geometries: A list of geometries from 'pina.geometry' 
            such as 'EllipsoidDomain' or 'CartesianDomain'. The first 
            geometry in the list is the geometry from which points are
            sampled. The rest of the geometries are the geometries that
            are excluded from the first geometry to find the difference.

        :Example:
            # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})

            # Create a Difference of the ellipsoid domains
            >>> difference = Difference([ellipsoid1, ellipsoid2])
        """
        super().__init__(geometries)

    def is_inside(self, point, check_border=False):
        for geometry in self.geometries[1:]:
            if geometry.is_inside(point):
                return False
        return self.geometries[0].is_inside(point, check_border)
    
    def sample(self, n, mode='random', variables='all'):
        """Sample routine for difference domain.

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

            # Create a Difference of the ellipsoid domains
            >>> difference = Difference([cartesian1, cartesian2])

            >>> difference.sample(n=5)
                LabelTensor([[0.8400, 0.9179],
                            [0.9154, 0.5769],
                            [1.7403, 0.4835],
                            [0.9545, 1.2851],
                            [1.3726, 0.9831]])

            >>> len(difference.sample(n=5)
                5

        """
        if mode != 'random':
            raise NotImplementedError(
                f'{mode} is not a valid mode for sampling.')

        sampled = []

        # sample the points
        while len(sampled) < n:
            # get sample point from first geometry
            point = self.geometries[0].sample(1, mode, variables)
            is_inside = False

            # check if point is inside any other geometry
            for geometry in self.geometries[1:]:
                # if point is inside any other geometry, break
                if geometry.is_inside(point):
                    is_inside = True
                    break
            # if point is not inside any other geometry, add to sampled
            if not is_inside:
                sampled.append(point)

        return LabelTensor(torch.cat(sampled), labels=self.variables)
