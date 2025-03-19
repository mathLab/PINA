"""Old module for the PINNs solver. Deprecated in 0.2.0."""

import warnings

from ...solver.physics_informed_solver import *
from ...utils import custom_warning_format

# back-compatibility 0.1
# Set the custom format for warnings
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=DeprecationWarning)
warnings.warn(
    "'pina.solvers.pinns' is deprecated and will be removed "
    "in future versions. Please use "
    "'pina.solver.physics_informed_solver' instead.",
    DeprecationWarning,
)
