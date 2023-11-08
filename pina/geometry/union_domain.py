"""Module for Union class. """

import torch
from .operation_interface import OperationInterface
from ..label_tensor import LabelTensor
import random


class Union(OperationInterface):

    def __init__(self, geometries):
        r"""
        PINA implementation of Unions of Domains.
        Given two sets :math:`A` and :math:`B` then the
        domain difference is defined as:

        .. math::
            A \cup B = \{x \mid x \in A \lor x \in B\},

        with :math:`x` a point in :math:`\mathbb{R}^N` and :math:`N`
        the dimension of the geometry space.

        :param list geometries: A list of geometries from ``pina.geometry`` 
            such as ``EllipsoidDomain`` or ``CartesianDomain``.

        :Example:
            >>> # Create two ellipsoid domains
            >>> ellipsoid1 = EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]})
            >>> ellipsoid2 = EllipsoidDomain({'x': [0, 2], 'y': [0, 2]})
            >>> # Create a union of the ellipsoid domains
            >>> union = GeometryUnion([ellipsoid1, ellipsoid2])

        """
        super().__init__(geometries)

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the ``Union`` domain.

        :param point: Point to be checked.
        :type point: LabelTensor
        :param check_border: Check if the point is also on the frontier
            of the ellipsoid, default ``False``.
        :type check_border: bool
        :return: Returning ``True`` if the point is inside, ``False`` otherwise.
        :rtype: bool
        """
        for geometry in self.geometries:
            if geometry.is_inside(point, check_border):
                return True
        return False

    def sample(self, n, mode='random', variables='all'):
        """
        Sample routine for ``Union`` domain.

        :param int n: Number of points to sample in the shape.
        :param str mode: Mode for sampling, defaults to ``random``. Available modes include: ``random``.
        :param variables: Variables to be sampled, defaults to ``all``.
        :type variables: str | list[str]
        :return: Returns ``LabelTensor`` of n sampled points.
        :rtype: LabelTensor

        :Example:
            >>> # Create two ellipsoid domains
            >>> cartesian1 = CartesianDomain({'x': [0, 2], 'y': [0, 2]})
            >>> cartesian2 = CartesianDomain({'x': [1, 3], 'y': [1, 3]})
            >>> # Create a union of the ellipsoid domains
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

        # calculate the number of points to sample for each geometry and the remainder
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
                geometry.sample(num_points + int(i < remainder), mode,
                                variables))
            # in case number of sampled points is smaller than the number of geometries
            if len(sampled_points) >= n:
                break

        return LabelTensor(torch.cat(sampled_points), labels=self.variables)
