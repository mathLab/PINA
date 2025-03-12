"""Module for the Union Operation."""

import random
import torch
from .operation_interface import OperationInterface
from ..label_tensor import LabelTensor


class Union(OperationInterface):
    r"""
    Implementation of the union operation between of a list of domains.

    Given two sets :math:`A` and :math:`B`, define the union of the two sets as:

    .. math::
        A \cup B = \{x \mid x \in A \lor x \in B\},

    where :math:`x` is a point in :math:`\mathbb{R}^N`.
    """

    def __init__(self, geometries):
        """
        Initialization of the :class:`Union` class.

        :param list[DomainInterface] geometries: A list of instances of the
            :class:`~pina.domain.DomainInterface` class on which the union
            operation is performed.

        :Example:
            >>> # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})
            >>> # Define the union of the domains
            >>> union = Union([ellipsoid1, ellipsoid2])
        """
        super().__init__(geometries)

    @property
    def sample_modes(self):
        """
        List of available sampling modes.
        """
        self.sample_modes = list(
            set(geom.sample_modes for geom in self.geometries)
        )

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the resulting domain.

        :param LabelTensor point: Point to be checked.
        :param bool check_border: If ``True``, the border is considered inside
            the domain. Default is ``False``.
        :return: ``True`` if the point is inside the domain, ``False`` otherwise.
        :rtype: bool
        """
        for geometry in self.geometries:
            if geometry.is_inside(point, check_border):
                return True
        return False

    def sample(self, n, mode="random", variables="all"):
        """
        Sampling routine.

        :param int n: Number of points to sample.
        :param str mode: Sampling method. Default is ``random``.
            Available modes: random sampling, ``random``;
        :param list[str] variables: variables to be sampled. Default is ``all``.
        :return: Sampled points.
        :rtype: LabelTensor

        :Example:
            >>> # Create two cartesian domains
            >>> cartesian1 = CartesianDomain({'x': [0, 2], 'y': [0, 2]})
            >>> cartesian2 = CartesianDomain({'x': [1, 3], 'y': [1, 3]})
            >>> # Define the union of the domains
            >>> union = Union([cartesian1, cartesian2])
            >>> # Sample
            >>> union.sample(n=5)
                LabelTensor([[1.2128, 2.1991],
                            [1.3530, 2.4317],
                            [2.2562, 1.6605],
                            [0.8451, 1.9878],
                            [1.8623, 0.7102]])
            >>> len(union.sample(n=5)
                5
        """
        sampled_points = []

        # calculate the number of points to sample for each geometry and the
        # remainder
        remainder = n % len(self.geometries)
        num_points = n // len(self.geometries)

        # sample the points
        # NB. geometries as shuffled since if we sample
        # multiple times just one point, we would end
        # up sampling only from the first geometry.
        iter_ = random.sample(self.geometries, len(self.geometries))
        for i, geometry in enumerate(iter_):
            # int(i < remainder) is one only if we have a remainder
            # different than zero. Notice that len(geometries) is
            # always smaller than remaider.
            sampled_points.append(
                geometry.sample(
                    num_points + int(i < remainder), mode, variables
                )
            )
            # in case number of sampled points is smaller than the number of
            # geometries
            if len(sampled_points) >= n:
                break

        return LabelTensor(torch.cat(sampled_points), labels=self.variables)
