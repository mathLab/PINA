.. docmeta::
   :last_reviewed: 2026-06-24


LabelTensor
===========

A :class:`~pina._src.core.label_tensor.LabelTensor` extends :class:`torch.Tensor` with named labels for each dimension.
This allows intuitive indexing by name (e.g. ``tensor.labels``) and is the fundamental data structure used throughout PINA
for passing multi-physics solution fields between components.

Use a :class:`~pina._src.core.label_tensor.LabelTensor` whenever you need to keep track of which
variable corresponds to which column in a tensor — for example, when a model outputs both temperature and pressure,
or when a domain has spatial coordinates ``x``, ``y``, ``z``.

.. currentmodule:: pina.label_tensor

.. automodule:: pina._src.core.label_tensor
   :no-members:

.. autoclass:: pina._src.core.label_tensor.LabelTensor
   :members:
   :private-members:
   :show-inheritance:

See Also
--------

* :class:`~pina.graph.Graph`
* :doc:`Data module <../_rst/data/data_module>`
