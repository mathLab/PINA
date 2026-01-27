"""Runtime type enforcement and validation utilities.

This module provides decorators to validate function arguments against type
annotations at runtime. This ensures that PINA components receive inputs
adhering to expected specifications, improving the robustness of the
scientific computing pipeline.
"""

from pina._src.core.type_checker import enforce_types

__all__ = ["enforce_types"]
