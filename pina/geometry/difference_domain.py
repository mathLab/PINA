"""Module for Location class."""
import torch
from .location import Location
from ..utils import check_consistency
from ..label_tensor import LabelTensor


class Difference(Location):
    """ PINA implementation of Difference of Domains."""

    def __init__(self, geometries):
        """ PINA implementation of Difference of Domains.

        :param list geometries: A list of geometries from 'pina.geometry' 
            such as 'EllipsoidDomain' or 'CartesianDomain'.

        :Example:
            # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})

            # Create a difference of the ellipsoid domains
            >>> difference = Difference([ellipsoid1, ellipsoid2])
        """
        super().__init__()

        # difference checks
        check_consistency(geometries, Location)
        self._check_difference_dimesions(geometries)

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
        """Check if a point is inside the difference domain.

        :param point: Point to be checked.
        :type point: torch.Tensor   
        :param bool check_border: If True, the border is considered inside.
        :return: True if the point is inside the difference domain, False otherwise.
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

            # Create a difference of the ellipsoid domains
            >>> difference = Difference([cartesian1, cartesian2])

            >>> difference.sample(n=1000)
                LabelTensor([[1.6802, 0.7723],
                    [1.8872, 0.9972],
                    [0.3166, 0.1213],
                    ...,
                    [2.1999, 2.0767],
                    [1.0655, 2.7711],
                    [1.6751, 2.3695]])

            >>> len(difference.sample(n=1000)
                1000

        """
        sampled = []

        # calculate the number of points to sample for each geometry and the remainder
        remainder = n % len(self.geometries)
        num_points = n // len(self.geometries)

        # sample the points
        for i, geometry in enumerate(self.geometries):
            if i < remainder:
                num_points += 1
            sampled_points = []
            # makes sure point is uniquely inside 1 shape
            while len(sampled_points) < num_points:
                sample = geometry.sample(1, 'random')
                # if not self.is_inside(sample) --> will be the intersection
                if self.is_inside(sample):
                    sampled_points.append(sample)
            sampled.extend(sampled_points)

        tensors = [point.data for point in sampled]
        concatenated_tensor = torch.cat(tensors, dim=0)

        return LabelTensor(concatenated_tensor, labels=[f'{i}' for i in self.variables])

    def _check_difference_dimesions(self, geometries):
        """Check if the dimensions of the geometries are consistent.

        :param geometries: Geometries to be checked.
        :type geometries: list[Location]
        """
        for geometry in geometries:
            if geometry.variables != geometries[0].variables:
                raise NotImplementedError(
                    f'The geometries need to be the same dimensions. {geometry.variables} is not equal to {geometries[0].variables}.')
