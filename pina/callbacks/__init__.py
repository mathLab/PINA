"""
Old module for callbacks. Deprecated in 0.2.0.
"""

import warnings

from ..callback import *
from ..utils import custom_warning_format

# back-compatibility 0.1
# Set the custom format for warnings
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=DeprecationWarning)
warnings.warn(
    "'pina.callbacks' is deprecated and will be removed "
    "in future versions. Please use 'pina.callback' instead.",
    DeprecationWarning,
)
