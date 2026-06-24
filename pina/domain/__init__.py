"""Geometry and domain definitions for spatial sampling.

This module provides tools for defining the physical space of a problem,
including primitive shapes (Cartesian, Ellipsoid, Simplex) and set-theoretic
operations (Union, Intersection, etc.) for building complex geometries.

:Example:

    >>> from pina.domain import CartesianDomain, EllipsoidDomain, Union
    >>> left = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
    >>> right = CartesianDomain({'x': [2, 3], 'y': [0, 1]})
    >>> domain = Union([left, right])
"""

__all__ = [
    "DomainInterface",
    "BaseDomain",
    "CartesianDomain",
    "EllipsoidDomain",
    "SimplexDomain",
    "OperationInterface",
    "Union",
    "Intersection",
    "Difference",
    "Exclusion",
]

from pina._src.domain.domain_interface import DomainInterface
from pina._src.domain.base_domain import BaseDomain
from pina._src.domain.cartesian_domain import CartesianDomain
from pina._src.domain.ellipsoid_domain import EllipsoidDomain
from pina._src.domain.simplex_domain import SimplexDomain
from pina._src.domain.operation_interface import OperationInterface
from pina._src.domain.union import Union
from pina._src.domain.intersection import Intersection
from pina._src.domain.difference import Difference
from pina._src.domain.exclusion import Exclusion
