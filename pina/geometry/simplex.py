import torch
from .location import Location
from pina.geometry import CartesianDomain
from pina import LabelTensor
from ..utils import check_consistency


class SimplexDomain(Location):
    """PINA implementation of a Simplex."""

    def __init__(self, simplex_matrix, sample_surface=False):
        """
        :param simplex_matrix: A matrix of LabelTensor objects representing
            a vertex of the simplex (a tensor), and the coordinates of the
            point (a list of labels).

        :type simplex_matrix: list[LabelTensor]
        :param sample_surface: A variable for choosing sample strategies. If
            ``sample_surface=True`` only samples on the Simplex surface
            frontier are taken. If ``sample_surface=False``, no such criteria
            is followed.

        :type sample_surface: bool

        .. warning::
            Sampling for dimensions greater or equal to 10 could result
            in a shrinking of the simplex, which degrades the quality
            of the samples. For dimensions higher than 10, other algorithms
            for sampling should be used.

        :Example:
            >>> spatial_domain = SimplexDomain(
                    [
                        LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
                        LabelTensor(torch.tensor([[1, 1]]), labels=["x", "y"]),
                        LabelTensor(torch.tensor([[0, 2]]), labels=["x", "y"]),
                    ], sample_surface = True
                )
        """

        # check consistency of sample_surface as bool
        check_consistency(sample_surface, bool)
        self._sample_surface = sample_surface

        # check consistency of simplex_matrix as list or tuple
        check_consistency([simplex_matrix], (list, tuple))

        # check everything within simplex_matrix is a LabelTensor
        check_consistency(simplex_matrix, LabelTensor)

        # check consistency of labels
        matrix_labels = simplex_matrix[0].labels
        if not all(vertex.labels == matrix_labels for vertex in simplex_matrix):
            raise ValueError(f"Labels don't match.")

        # check consistency dimensions
        dim_simplex = len(matrix_labels)
        if len(simplex_matrix) != dim_simplex + 1:
            raise ValueError(
                "An n-dimensional simplex is composed by n + 1 tensors of dimension n."
            )

        # creating vertices matrix
        self._vertices_matrix = LabelTensor.vstack(simplex_matrix)

        # creating basis vectors for simplex
        # self._vectors_shifted = (
        #     (self._vertices_matrix.T)[:, :-1] - (self._vertices_matrix.T)[:, None, -1]
        # )  ### TODO: Remove after checking

        vert = self._vertices_matrix
        self._vectors_shifted = (vert[:-1] - vert[-1]).T

        # build cartesian_bound
        self._cartesian_bound = self._build_cartesian(self._vertices_matrix)

    @property
    def variables(self):
        return self._vertices_matrix.labels

    def _build_cartesian(self, vertices):
        """
        Build Cartesian border for Simplex domain to be used in sampling.
        :param vertex_matrix: matrix of vertices
        :type vertices: list[list]
        :return: Cartesian border for triangular domain
        :rtype: CartesianDomain
        """

        span_dict = {}

        for i, coord in enumerate(self.variables):
            sorted_vertices = sorted(vertices, key=lambda vertex: vertex[i])
            # respective coord bounded by the lowest and highest values
            span_dict[coord] = [
                float(sorted_vertices[0][i]),
                float(sorted_vertices[-1][i])
            ]

        return CartesianDomain(span_dict)

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the simplex.
        Uses the algorithm described involving barycentric coordinates:
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system.

        :param point: Point to be checked.
        :type point: LabelTensor
        :param check_border: Check if the point is also on the frontier
            of the simplex, default ``False``.
        :type check_border: bool
        :return: Returning ``True`` if the point is inside, ``False`` otherwise.
        :rtype: bool

        .. note::
            When ``sample_surface`` in the ``__init()__``
            is set to ``True``, then the method only checks
            points on the surface, and not inside the domain.
        """

        if not all(label in self.variables for label in point.labels):
            raise ValueError("Point labels different from constructor"
                             f" dictionary labels. Got {point.labels},"
                             f" expected {self.variables}.")

        point_shift = point - self._vertices_matrix[-1]
        point_shift = point_shift.tensor.reshape(-1, 1)

        # compute barycentric coordinates
        lambda_ = torch.linalg.solve(self._vectors_shifted * 1.0,
                                     point_shift * 1.0)
        lambda_1 = 1.0 - torch.sum(lambda_)
        lambdas = torch.vstack([lambda_, lambda_1])

        # perform checks
        if not check_border:
            return all(torch.gt(lambdas, 0.0)) and all(torch.lt(lambdas, 1.0))

        return all(torch.ge(lambdas, 0)) and (any(torch.eq(lambdas, 0))
                                              or any(torch.eq(lambdas, 1)))

    def _sample_interior_randomly(self, n, variables):
        """
        Randomly sample points inside a simplex of arbitrary
        dimension, without the boundary.
        :param int n: Number of points to sample in the shape.
        :param variables: pinn variable to be sampled, defaults to ``all``.
        :type variables: str or list[str], optional
        :return: Returns tensor of n sampled points.
        :rtype: torch.Tensor
        """

        # =============== For Developers ================ #
        #
        # The sampling startegy used is fairly simple.
        # First we sample a random vector from the hypercube
        # which contains the simplex. Then, if the point
        # sampled is inside the simplex, we add it as a valid
        # one.
        #
        # =============================================== #

        sampled_points = []
        while len(sampled_points) < n:
            sampled_point = self._cartesian_bound.sample(n=1,
                                                         mode="random",
                                                         variables=variables)

            if self.is_inside(sampled_point, self._sample_surface):
                sampled_points.append(sampled_point)
        return torch.cat(sampled_points, dim=0)

    def _sample_boundary_randomly(self, n):
        """
        Randomly sample points on the boundary of a simplex
        of arbitrary dimensions.
        :param int n: Number of points to sample in the shape.
        :return: Returns tensor of n sampled points
        :rtype: torch.Tensor
        """

        # =============== For Developers ================ #
        #
        # The sampling startegy used is fairly simple.
        # We first sample the lambdas in [0, 1] domain,
        # we then set to zero only one lambda, and normalize.
        # Finally, we compute the matrix product between the
        # lamdas and the vertices matrix.
        #
        # =============================================== #

        sampled_points = []

        while len(sampled_points) < n:
            # extract number of vertices
            number_of_vertices = self._vertices_matrix.shape[0]
            # extract idx lambda to set to zero randomly
            idx_lambda = torch.randint(low=0,
                                       high=number_of_vertices,
                                       size=(1, ))
            # build lambda vector
            # 1. sampling [1, 2)
            lambdas = torch.rand((number_of_vertices, 1))
            # 2. setting lambdas[idx_lambda] to 0
            lambdas[idx_lambda] = 0
            # 3. normalize
            lambdas /= lambdas.sum()
            # 4. compute dot product
            sampled_points.append(self._vertices_matrix.T @ lambdas)
        return torch.cat(sampled_points, dim=1).T

    def sample(self, n, mode="random", variables="all"):
        """
        Sample n points from Simplex domain.

        :param int n: Number of points to sample in the shape.
        :param str mode: Mode for sampling, defaults to ``random``. Available modes include: ``random``.
        :param variables: Variables to be sampled, defaults to ``all``.
        :type variables: str | list[str]
        :return: Returns ``LabelTensor`` of n sampled points.
        :rtype: LabelTensor

        .. warning::
            When ``sample_surface = True`` in the initialization, all
            the variables are sampled, despite passing different once
            in ``variables``.
        """

        if mode in ["random"]:
            if self._sample_surface:
                sample_pts = self._sample_boundary_randomly(n)
            else:
                sample_pts = self._sample_interior_randomly(n, variables)

        else:
            raise NotImplementedError(f"mode={mode} is not implemented.")

        return LabelTensor(sample_pts, labels=self.variables)