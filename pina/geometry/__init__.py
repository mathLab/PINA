"""Old module for geometry classes and functions. Deprecated in 0.2.0."""

import warnings

from ..domain import *
from ..utils import custom_warning_format

# back-compatibility 0.1
# creating alias
Location = DomainInterface

# Set the custom format for warnings
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=DeprecationWarning)
warnings.warn(
    "'pina.geometry' is deprecated and will be removed "
    "in future versions. Please use 'pina.domain' instead. "
    "Location moved to DomainInferface object.",
    DeprecationWarning,
)
