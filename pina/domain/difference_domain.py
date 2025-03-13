"""Module for the Difference Operation."""

import torch
from .operation_interface import OperationInterface
from ..label_tensor import LabelTensor


class Difference(OperationInterface):
    r"""
    Implementation of the difference operation between of a list of domains.

    Given two sets :math:`A` and :math:`B`, define the difference of the two
    sets as:

    .. math::
        A - B = \{x \mid x \in A \land x \not\in B\},

    where :math:`x` is a point in :math:`\mathbb{R}^N`.
    """

    def __init__(self, geometries):
        """
        Initialization of the :class:`Difference` class.

        :param list[DomainInterface] geometries: A list of instances of the
            :class:`~pina.domain.domain_interface.DomainInterface` class on
            which the difference operation is performed. The first domain in the
            list serves as the base from which points are sampled, while the
            remaining domains define the regions to be excluded from the base
            domain to compute the difference.

        :Example:
            >>> # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})
            >>> # Define the difference between the domains
            >>> difference = Difference([ellipsoid1, ellipsoid2])
        """
        super().__init__(geometries)

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the resulting domain.

        :param LabelTensor point: Point to be checked.
        :param bool check_border: If ``True``, the border is considered inside
            the domain. Default is ``False``.
        :return: ``True`` if the point is inside the domain,
            ``False`` otherwise.
        :rtype: bool
        """
        for geometry in self.geometries[1:]:
            if geometry.is_inside(point):
                return False
        return self.geometries[0].is_inside(point, check_border)

    def sample(self, n, mode="random", variables="all"):
        """
        Sampling routine.

        :param int n: Number of points to sample.
        :param str mode: Sampling method. Default is ``random``.
            Available modes: random sampling, ``random``;
        :param list[str] variables: variables to be sampled. Default is ``all``.
        :raises NotImplementedError: If the sampling method is not implemented.
        :return: Sampled points.
        :rtype: LabelTensor

        :Example:
            >>> # Create two Cartesian domains
            >>> cartesian1 = CartesianDomain({'x': [0, 2], 'y': [0, 2]})
            >>> cartesian2 = CartesianDomain({'x': [1, 3], 'y': [1, 3]})
            >>> # Define the difference between the domains
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
