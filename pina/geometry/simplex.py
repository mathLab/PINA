import torch
from .location import Location
from pina.geometry import CartesianDomain
from pina import LabelTensor
from ..utils import check_consistency


class SimplexDomain(Location):
    """PINA implementation of a Simplex."""

    def __init__(self, simplex_dict, labels, sample_surface=False):
        """
        :param simplex_dict: A dictionary with dict-key a string representing
            the input variables for the problem, and dict-value a list
            representing vertices of the simplex.
        :type simplex_dict: dict
        :param labels: A list of labels for vertex components. Represents the
            order in which points should list coordinates.
        :type labels: list[str]
        :param sample_surface: A variable for choosing sample strategies. If
            `sample_surface=True` only samples on the Simplex surface
            frontier are taken. If `sample_surface=False`, no such criteria
            is followed.
        :type sample_surface: bool

        .. warning::
            Sampling for dimensions greater or equal to 10 could result
            in a shrinking of the simplex, which degrades the quality
            of the samples. For dimensions higher than 10, other algorithms
            for sampling should be used.

        :Example:

            >>> spatial_domain = SimplexDomain({'vertex1': [0, 0], 
                                                'vertex2': [1, 1], 
                                                'vertex3': [0, 2]}, 
                                                ['x', 'y'])
        """

        # check consistency of labels
        if not isinstance(labels, list):
            raise ValueError(f"{type(labels).__name__} must be {list}.")
        check_consistency(labels, str)
        self._variables = labels

        # check consistency of sample_surface
        check_consistency(sample_surface, bool)
        self._sample_surface = sample_surface

        # check consistency of simplex_dict
        check_consistency(simplex_dict, dict)
        for vertex in simplex_dict.values():
            if not isinstance(vertex, list):
                raise ValueError(f"{type(vertex).__name__} must be {list}.")

        # vertices, vectors, dimension
        self._vertices_matrix = torch.tensor(list(simplex_dict.values()), dtype=torch.float).T
        self._vectors_shifted = self._vertices_matrix[:, :-1] - self._vertices_matrix[:, None, -1]

        # build cartesian_bound
        self._cartesian_bound = self._build_cartesian(
            list(simplex_dict.values())
        )

    @property
    def variables(self):
        return self._variables

    def _build_cartesian(self, vertices):
        """
        Build Cartesian border for Simplex domain to be used in sampling.

        :param vertices: list of Simplex domain's vertices
        :type vertices: list[list]
        :return: Cartesian border for triangular domain
        :rtype: CartesianDomain
        """

        span_dict = {}

        for i, coord in enumerate(self.variables):
            sorted_vertices = sorted(vertices, key=lambda vertex: vertex[i])

            # respective coord bounded by the lowest and highest values
            span_dict[coord] = [sorted_vertices[0][i], sorted_vertices[-1][i]]

        return CartesianDomain(span_dict)

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the simplex.
        Uses the algorithm described involving barycentric coordinates:
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system

        .. note::
            When ```'sample_surface'``` in the ```'__init()__'```
            is set to ```'True'```, then the method only checks 
            points on the surface, and not inside the domain.

        :param point: Point to be checked.
        :type point: LabelTensor
        :param check_border: Check if the point is also on the frontier
            of the simplex, default False.
        :type check_border: bool
        :return: Returning True if the point is inside, False otherwise.
        :rtype: bool
        """

        if not all([label in self.variables for label in point.labels]):
            raise ValueError(
                "Point labels different from constructor"
                f" dictionary labels. Got {point.labels},"
                f" expected {self.variables}."
            )

        # shift point
        point_shift = point.T - self._vertices_matrix[:, None, -1]

        # compute barycentric coordinates
        lambda_ = torch.linalg.solve(self._vectors_shifted, point_shift)
        lambda_1 = 1.0 - torch.sum(lambda_)
        lambdas = torch.vstack([lambda_, lambda_1])

        # perform checks
        if not check_border:
            return all(torch.gt(lambdas, 0.)) and all(torch.lt(lambdas, 1.))

        return all(torch.ge(lambdas, 0)) and (
            any(torch.eq(lambdas, 0)) or any(torch.eq(lambdas, 1))
        )

    def sample(self, n, mode="random", variables="all"):
        """
        Sample n points from Simplex domain.

        :param n: Number of points to sample in the shape.
        :type n: int
        :param mode: Mode for sampling, defaults to 'random'.
            Available modes include: 'random'.
        :type mode: str, optional
        :param variables: pinn variable to be sampled, defaults to 'all'.
        :type variables: str or list[str], optional
        :return: Returns LabelTensor of n sampled points
        :rtype: LabelTensor(tensor)
        """

        if mode in ["random"]:
            # Sample points on the domain
            sampled_points = []
            
            while len(sampled_points) < n:
                sampled_point = self._cartesian_bound.sample(
                    n=1, mode="random", variables=variables
                )

                if self.is_inside(sampled_point, self._sample_surface):
                    sampled_points.append(sampled_point)
        
        else:
            raise NotImplementedError(f'mode={mode} is not implemented.')

        return LabelTensor(torch.cat(sampled_points, dim=0), labels=self.variables)
