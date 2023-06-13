import torch
from .location import Location


class Union(Location):
    """ PINA implementation of Unions of Domains."""

    def __init__(self, geometries):
        """ PINA implementation of Unions of Domains.

        :param geometries: A list of shapes from pina.geometry such as 
                EllipsoidDomain or TriangleDomain

        :Example:
            # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})

            # Create a union of the ellipsoid domains
            >>> union = GeometryUnion([ellipsoid1, ellipsoid2])

        """
        super().__init__()
        self.geometries = geometries

    @property
    def varibles(self):
        """
        Spatial variables.

        :return: All the spatial variables defined in '__init__()'
        :rtype: list[str]
        """
        all_variables = set()
        for geometry in self.geometries:
            all_variables.update(geometry.variables)
        return list(all_variables)

    def is_inside(self, point, check_border=False):
        """Check if a point is inside the union domain.

        :param point: Point to be checked
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
        for geometry in self.geometries:
            points = geometry.sample(
                int(n/len(self.geometries)), mode, variables)
            sampled_points.append(points)

        combined_points = torch.cat(sampled_points)

        return combined_points
