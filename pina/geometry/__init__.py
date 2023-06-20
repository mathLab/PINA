__all__ = [
    'Location',
    'CartesianDomain',
    'EllipsoidDomain',
    'Union', 
    'TriangleDomain'
]

from .location import Location
from .cartesian import CartesianDomain
from .ellipsoid import EllipsoidDomain
from .exclusion_domain import Exclusion
from .intersection_domain import Intersection
from .union_domain import Union
from .difference_domain import Difference
from .union_domain import Union
<<<<<<< HEAD
from .triangle import SimplexDomain
=======
from .triangle import TriangleDomain
>>>>>>> 053d570 (Triangle geometry)
