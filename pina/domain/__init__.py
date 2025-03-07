"""
This module contains the domain classes.
"""

__all__ = [
    "DomainInterface",
    "CartesianDomain",
    "EllipsoidDomain",
    "Union",
    "Intersection",
    "Exclusion",
    "Difference",
    "OperationInterface",
    "SimplexDomain",
]

from .domain_interface import DomainInterface
from .cartesian import CartesianDomain
from .ellipsoid import EllipsoidDomain
from .exclusion_domain import Exclusion
from .intersection_domain import Intersection
from .union_domain import Union
from .difference_domain import Difference
from .operation_interface import OperationInterface
from .simplex import SimplexDomain
