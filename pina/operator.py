"""A public API for differential operators and automatic differentiation utilities.

This module provides standard vector calculus operators (gradient, divergence,
laplacian, advection) implemented using automatic differentiation. It includes
both high-level general operators and optimized 'fast' variants for improved
computational efficiency during training.

:Example:

    >>> from pina.operator import grad, div, laplacian
    >>> import torch
    >>> x = torch.rand(10, 2, requires_grad=True)
    >>> f = x.pow(2).sum(dim=1, keepdim=True)
    >>> g = grad(f, x)
    >>> g.shape
    torch.Size([10, 2])
"""

from pina._src.core.operator import (
    grad,
    fast_grad,
    fast_div,
    fast_laplacian,
    fast_advection,
    div,
    laplacian,
    advection,
)

__all__ = [
    "grad",
    "fast_grad",
    "fast_div",
    "fast_laplacian",
    "fast_advection",
    "div",
    "laplacian",
    "advection",
]
