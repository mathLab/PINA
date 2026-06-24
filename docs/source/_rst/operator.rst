.. docmeta::
   :last_reviewed: 2026-06-24


Operators
=========

Differential operators are used in PINA to compute spatial and temporal derivatives of neural network outputs.
They are essential for defining PDE residuals in PINN-based solvers.

The following operators are available:

* :func:`~pina.operator.grad` — Gradient of a scalar field.
* :func:`~pina.operator.div` — Divergence of a vector field.
* :func:`~pina.operator.laplacian` — Laplacian of a scalar field.
* :func:`~pina.operator.advection` — Advection operator.
* ``fast_grad``, ``fast_div``, ``fast_laplacian``, ``fast_advection`` — Optimised variants for performance.

.. currentmodule:: pina.operator

.. automodule:: pina._src.core.operator
   :members:
   :show-inheritance:

.. note::

   All operators work with :class:`~pina.label_tensor.LabelTensor` inputs and respect label-based indexing.

See Also
--------

* :class:`~pina.equation.equation.Equation`
* :class:`~pina.equation.equation_factory.Poisson`
