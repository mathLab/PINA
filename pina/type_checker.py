"""Runtime type enforcement and validation utilities.

This module provides decorators to validate function arguments against type
annotations at runtime. This ensures that PINA components receive inputs
adhering to expected specifications, improving the robustness of the
scientific computing pipeline.

:Example:

    >>> from pina.type_checker import enforce_types
    >>> @enforce_types
    ... def compute(x: float, y: int) -> float:
    ...     return x ** y
"""

from pina._src.core.type_checker import enforce_types

__all__ = ["enforce_types"]
