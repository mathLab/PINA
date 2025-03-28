"""Module for the Simplex Domain."""

import torch
from .domain_interface import DomainInterface
from .cartesian import CartesianDomain
from ..label_tensor import LabelTensor
from ..utils import check_consistency


class SimplexDomain(DomainInterface):
    """
    Implementation of the simplex domain.
    """

    def __init__(self, simplex_matrix, sample_surface=False):
        """
        Initialization of the :class:`SimplexDomain` class.

        :param list[LabelTensor] simplex_matrix: A matrix representing the
            vertices of the simplex.
        :param bool sample_surface: A flag to choose the sampling strategy.
            If ``True``, samples are taken only from the surface of the simplex.
            If ``False``, samples are taken from the interior of the simplex.
            Default is ``False``.
        :raises ValueError: If the labels of the vertices don't match.
        :raises ValueError: If the number of vertices is not equal to the
            dimension of the simplex plus one.

        .. warning::
            Sampling for dimensions greater or equal to 10 could result in a
            shrinkage of the simplex, which degrades the quality of the samples.
            For dimensions higher than 10, use other sampling algorithms.

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
            raise ValueError("Labels don't match.")

        # check consistency dimensions
        dim_simplex = len(matrix_labels)
        if len(simplex_matrix) != dim_simplex + 1:
            raise ValueError(
                "An n-dimensional simplex is composed by n + 1 tensors of "
                "dimension n."
            )

        # creating vertices matrix
        self._vertices_matrix = LabelTensor.vstack(simplex_matrix)

        # creating basis vectors for simplex
        vert = self._vertices_matrix
        self._vectors_shifted = (vert[:-1] - vert[-1]).T

        # build cartesian_bound
        self._cartesian_bound = self._build_cartesian(self._vertices_matrix)

    @property
    def sample_modes(self):
        """
        List of available sampling modes.

        :return: List of available sampling modes.
        :rtype: list[str]
        """
        return ["random"]

    @property
    def variables(self):
        """
        List of variables of the domain.

        :return: List of variables of the domain.
        :rtype: list[str]
        """
        return sorted(self._vertices_matrix.labels)

    def _build_cartesian(self, vertices):
        """
        Build the cartesian border for a simplex domain to be used in sampling.

        :param list[LabelTensor] vertices: list of vertices defining the domain.
        :return: The cartesian border for the simplex domain.
        :rtype: CartesianDomain
        """

        span_dict = {}
        for coord in self.variables:
            sorted_vertices = torch.sort(vertices[coord].tensor.squeeze())
            # respective coord bounded by the lowest and highest values
            span_dict[coord] = [
                float(sorted_vertices.values[0]),
                float(sorted_vertices.values[-1]),
            ]

        return CartesianDomain(span_dict)

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the simplex. It uses an algorithm involving
        barycentric coordinates.

        :param LabelTensor point: Point to be checked.
        :param check_border: If ``True``, the border is considered inside
            the simplex. Default is ``False``.
        :raises ValueError: If the labels of the point are different from those
            passed in the ``__init__`` method.
        :return: ``True`` if the point is inside the domain,
            ``False`` otherwise.
        :rtype: bool
        """

        if not all(label in self.variables for label in point.labels):
            raise ValueError(
                "Point labels different from constructor"
                f" dictionary labels. Got {point.labels},"
                f" expected {self.variables}."
            )

        point_shift = point - self._vertices_matrix[-1]
        point_shift = point_shift.tensor.reshape(-1, 1)

        # compute barycentric coordinates
        lambda_ = torch.linalg.solve(
            self._vectors_shifted * 1.0, point_shift * 1.0
        )
        lambda_1 = 1.0 - torch.sum(lambda_)
        lambdas = torch.vstack([lambda_, lambda_1])

        # perform checks
        if not check_border:
            return all(torch.gt(lambdas, 0.0)) and all(torch.lt(lambdas, 1.0))

        return all(torch.ge(lambdas, 0)) and (
            any(torch.eq(lambdas, 0)) or any(torch.eq(lambdas, 1))
        )

    def _sample_interior_randomly(self, n, variables):
        """
        Sample at random points from the interior of the simplex. Boundaries are
        excluded from this sampling routine.

        :param int n: Number of points to sample.
        :param list[str] variables: variables to be sampled.
        :return: Sampled points.
        :rtype: list[torch.Tensor]
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
            sampled_point = self._cartesian_bound.sample(
                n=1, mode="random", variables=variables
            )

            if self.is_inside(sampled_point, self._sample_surface):
                sampled_points.append(sampled_point)
        return torch.cat(sampled_points, dim=0)

    def _sample_boundary_randomly(self, n):
        """
        Sample at random points from the boundary of the simplex.

        :param int n: Number of points to sample.
        :return: Sampled points.
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
            idx_lambda = torch.randint(
                low=0, high=number_of_vertices, size=(1,)
            )
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
        Sampling routine.

        :param int n: Number of points to sample.
        :param str mode: Sampling method. Default is ``random``.
            Available modes: random sampling, ``random``.
        :param list[str] variables: variables to be sampled. Default is ``all``.
        :raises NotImplementedError: If the sampling method is not implemented.
        :return: Sampled points.
        :rtype: LabelTensor

        .. warning::
            When ``sample_surface=True``, all variables are sampled,
            ignoring the ``variables`` parameter.
        """

        if variables == "all":
            variables = self.variables
        elif isinstance(variables, (list, tuple)):
            variables = sorted(variables)

        if mode in self.sample_modes:
            if self._sample_surface:
                sample_pts = self._sample_boundary_randomly(n)
            else:
                sample_pts = self._sample_interior_randomly(n, variables)

        else:
            raise NotImplementedError(f"mode={mode} is not implemented.")

        return LabelTensor(sample_pts, labels=self.variables)
