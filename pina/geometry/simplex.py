import torch

from pina.geometry import CartesianDomain
from .location import Location
from pina import LabelTensor
from ..utils import check_consistency


class SimplexDomain(Location):
    """PINA implementation of a Simplex."""

    def __init__(self, simplex_dict, labels, sample_surface=False):
        """
        :param simplex_dict: A dictionary with dict-key a string representing
            the input variables for the pinn, and dict-value a list
            representing vertices of the simplex.
        :type simplex_dict: dict
        :param labels: A list of labels for vertex components
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

            >>> spatial_domain = SimplexDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [0, 2]}, ['x', 'y'])
        """

        # check consistency of labels
        if not isinstance(labels, list):
            raise ValueError(f"{type(labels).__name__} must be {list}.")
        check_consistency(labels, str)
        self._coordinate_labels = labels

        # check consistency of sample_surface
        check_consistency(sample_surface, bool)
        self._sample_surface = sample_surface

        # check consistency of simplex_dict
        check_consistency(simplex_dict, dict)
        for vertex in simplex_dict.values():
            if not isinstance(vertex, list):
                raise ValueError(f"{type(vertex).__name__} must be {list}.")

        # vertices, vectors, dimension
        self._vertices = simplex_dict
        self._vert_list = torch.tensor(list(simplex_dict.values())).T
        self._vectors = (
            self._vertices_list[:, :-1] - self._vertices_list[:, None, -1]
        ).type(torch.FloatTensor)

        # build cartesian_bound
        self._cartesian_bound = self._build_cartesian(
            list(simplex_dict.values()), labels
        )

    @property
    def variables(self):
        """
        Coordinate labels of simplex.

        :return: Coordinate labels
        :rtype: list[str]
        """

        return self._coordinate_labels

    @property
    def vertices(self):
        """
        Vertices of simplex.

        :return: Vectors
        :rtype: list[list]
        """

        return self._vertices

    @property
    def vectors(self):
        """
        Vectors.

        :return: Vectors
        :rtype: dict(LabelTensor)
        """
        return self._vectors

    @property
    def _vertices_list(self):
        """
        List of vectors.

        :return: List of vectors
        :rtype: list[LabelTensor]
        """
        return self._vert_list

    @property
    def cartesian_bound(self):
        """
        Cartesian border for Simplex domain.

        :return: Cartesian border for Simplex domain
        :rtype: CartesianDomain
        """

        return self._cartesian_bound

    @property
    def sample_surface(self):
        """
        Whether the surface should be sampled or not.

        :return: Whether the surface should be sampled or not
        :rtype: bool
        """

        return self._sample_surface

    def _build_cartesian(self, vertices, labels):
        """
        Build Cartesian border for Simplex domain to be used in sampling.

        :param vertices: list of Simplex domain's vertices
        :type vertices: list[list]
        :return: Cartesian border for triangular domain
        :rtype: CartesianDomain
        """

        span_dict = {}

        for i, coord in enumerate(labels):
            sorted_vertices = sorted(vertices, key=lambda vertex: vertex[i])

            # respective coord bounded by the lowest and highest values
            span_dict[coord] = [sorted_vertices[0][i], sorted_vertices[-1][i]]

        return CartesianDomain(span_dict)

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the simplex.
        Uses the algorithm described here: 
        https://math.stackexchange.com/questions/1226707/how-to-check-if-point-x-in-mathbbrn-is-in-a-n-simplex

        :param point: Point to be checked
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

        point_shift = (point.T - self._vertices_list[:, None, -1]).type(
            torch.FloatTensor
        )
        lambda_ = torch.linalg.solve(self.vectors, point_shift)
        lambda_1 = 1.0 - torch.sum(lambda_)
        lambdas = torch.vstack([lambda_, lambda_1])

        if not check_border:
            return all(torch.gt(lambdas, 0)) and all(torch.lt(lambdas, 1))

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

        if mode != "random":
            raise ValueError("Mode can only be random")

        # Sample points on the domain
        sampled_points = []
        for _ in range(n):
            sampled_point = self.cartesian_bound.sample(
                n=1, mode="random", variables=variables
            )

            # Keep sampling until you get a point that is inside
            while not self.is_inside(sampled_point, self.sample_surface):
                sampled_point = self.cartesian_bound.sample(
                    n=1, mode=mode, variables=variables
                )

            sampled_points.append(sampled_point)

        return LabelTensor(torch.cat(sampled_points, dim=0), labels=self.variables)
