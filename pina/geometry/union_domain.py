import torch
from .location import Location
from .ellipsoid import EllipsoidDomain
from ..utils import check_consistency
from ..label_tensor import LabelTensor


class Union(Location):
    """ PINA implementation of Unions of Domains."""

    def __init__(self, geometries):
        """ PINA implementation of Unions of Domains.

        :param list geometries: A list of geometries from 'pina.geometry' 
            such as 'EllipsoidDomain' or 'CartesianDomain'.

        :Example:
            # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})

            # Create a union of the ellipsoid domains
            >>> union = GeometryUnion([ellipsoid1, ellipsoid2])

        """
        super().__init__()

        # union checks
        check_consistency(geometries, Location)
        self._check_union_dimensions(geometries)

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
        """Check if a point is inside the union domain.

        :param point: Point to be checked.
        :type point: LabelTensor
        :param check_border: Check if the point is also on the frontier
            of the ellipsoid, default False.
        :type check_border: bool
        :return: Returning True if the point is inside, False otherwise.
        :rtype: bool
        """
        for geometry in self.geometries:
            if geometry.is_inside(point, check_border):
                return True
        return False

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
            # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})

            # Create a union of the ellipsoid domains
            >>> union = GeometryUnion([ellipsoid1, ellipsoid2])

            >>> union.sample(n=1000)
                LabelTensor([[-0.2025,  0.0072],
                            [ 0.0358,  0.5748],
                            [ 0.5083,  0.0482],
                            ...,
                            [ 0.5857,  0.9279],
                            [ 1.1496,  1.7339],
                            [ 0.7650,  1.0469]])

            >>> len(union.sample(n=1000)
                1000
        """
        sampled_points = []

        # calculate the number of points to sample for each geometry and the remainder
        remainder = n % len(self.geometries)
        num_points = n // len(self.geometries)

        # sample the points
        for i, geometry in enumerate(self.geometries):
            # add to sample total if remainder is not 0
            if i < remainder:
                num_points += 1
            points = geometry.sample(num_points, mode, variables)
            sampled_points.append(points)

        return LabelTensor(torch.cat(sampled_points), labels=[f'{i}' for i in self.variables])
    

    def combine_domains(self):
        """Compute the union of the ellipsoids while preserving the outline."""
        # Extract the ellipsoid domains from the geometries
        ellipsoid_domains = [geometry for geometry in self.geometries if isinstance(
            geometry, EllipsoidDomain)]

        # Combine the ellipsoid domains
        if len(ellipsoid_domains) < 2:
            raise ValueError(
                "At least two EllipsoidDomain geometries are required for the union.")

        combined_domain = ellipsoid_domains[0]
        for domain in ellipsoid_domains[1:]:
            combined_domain = self._combine_ellipsoids(combined_domain, domain)

        return combined_domain

    def _combine_ellipsoids(self, domain1, domain2):
        """Combine two ellipsoid domains into a single domain."""
        # Compute the union of the fixed variables
        combined_fixed = {**domain1.fixed_, **domain2.fixed_}

        # Compute the union of the range variables
        combined_range = {**domain1.range_, **domain2.range_}

        # Create a new instance of the EllipsoidDomain class with the combined fixed and range variables
        union_domain = EllipsoidDomain({})
        union_domain.fixed_ = combined_fixed
        union_domain.range_ = combined_range

        return union_domain

    def _check_union_dimensions(self, geometries):
        """Check if the dimensions of the geometries are consistent.

        :param geometries: Geometries to be checked.
        :type geometries: list[Location]
        """
        for geometry in geometries:
            if geometry.variables != geometries[0].variables:
                raise NotImplementedError(
                    f'The geometries need to be the same dimensions. {geometry.variables} is not equal to {geometries[0].variables}')
