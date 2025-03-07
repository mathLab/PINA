"""Module for Difference class."""

import torch
from .operation_interface import OperationInterface
from ..label_tensor import LabelTensor


class Difference(OperationInterface):
    """
    PINA implementation of Difference of Domains.
    """

    # pylint: disable=W0246
    def __init__(self, geometries):
        r"""
        Given two sets :math:`A` and :math:`B` then the
        domain difference is defined as:

        .. math::
            A - B = \{x \mid x \in A \land x \not\in B\},

        with :math:`x` a point in :math:`\mathbb{R}^N` and :math:`N`
        the dimension of the geometry space.

        :param list geometries: A list of geometries from ``pina.geometry``
            such as ``EllipsoidDomain`` or ``CartesianDomain``. The first
            geometry in the list is the geometry from which points are
            sampled. The rest of the geometries are the geometries that
            are excluded from the first geometry to find the difference.

        :Example:
            >>> # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})
            >>> # Create a Difference of the ellipsoid domains
            >>> difference = Difference([ellipsoid1, ellipsoid2])
        """
        super().__init__(geometries)

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the ``Difference`` domain.

        :param point: Point to be checked.
        :type point: torch.Tensor
        :param bool check_border: If ``True``, the border is considered inside.
        :return: ``True`` if the point is inside the Exclusion domain,
            ``False`` otherwise.
        :rtype: bool
        """
        for geometry in self.geometries[1:]:
            if geometry.is_inside(point):
                return False
        return self.geometries[0].is_inside(point, check_border)

    def sample(self, n, mode="random", variables="all"):
        """
        Sample routine for ``Difference`` domain.

        :param int n: Number of points to sample in the shape.
        :param str mode: Mode for sampling, defaults to ``random``. Available
            modes include: ``random``.
        :param variables: Variables to be sampled, defaults to ``all``.
        :type variables: str | list[str]
        :return: Returns ``LabelTensor`` of n sampled points.
        :rtype: LabelTensor

        :Example:
            >>> # Create two Cartesian domains
            >>> cartesian1 = CartesianDomain({'x': [0, 2], 'y': [0, 2]})
            >>> cartesian2 = CartesianDomain({'x': [1, 3], 'y': [1, 3]})
            >>> # Create a Difference of the ellipsoid domains
            >>> difference = Difference([cartesian1, cartesian2])
            >>> # Sampling
            >>> difference.sample(n=5)
                LabelTensor([[0.8400, 0.9179],
                            [0.9154, 0.5769],
                            [1.7403, 0.4835],
                            [0.9545, 1.2851],
                            [1.3726, 0.9831]])
            >>> len(difference.sample(n=5)
                5

        """
        if mode not in self.sample_modes:
            raise NotImplementedError(
                f"{mode} is not a valid mode for sampling."
            )

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
