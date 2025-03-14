"""Old module for operators. Deprecated in 0.2.0."""

import warnings

from .operator import *
from .utils import custom_warning_format

# back-compatibility 0.1
# Set the custom format for warnings
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=DeprecationWarning)
warnings.warn(
    "'pina.operators' is deprecated and will be removed "
    "in future versions. Please use 'pina.operator' instead.",
    DeprecationWarning,
)
