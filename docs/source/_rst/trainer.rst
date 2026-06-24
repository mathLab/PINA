.. docmeta::
   :status: complete
   :needs_example: false
   :needs_advanced_example: true
   :reviewer:
   :last_reviewed: 2026-06-24

Trainer
=======

The :class:`~pina._src.core.trainer.Trainer` is the central orchestrator for model training in PINA.
It wraps the PyTorch Lightning Trainer and adds PINA-specific data handling, batching, and compilation support.

.. currentmodule:: pina.trainer

.. automodule:: pina._src.core.trainer
   :no-members:

.. autoclass:: pina._src.core.trainer.Trainer
   :members:
   :show-inheritance:

.. note::

   The Trainer accepts all keyword arguments from the
   `Lightning Trainer API <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`_.

See Also
--------

* :class:`~pina.solver.solver.SolverInterface`
* :class:`~pina.data.data_module.PinaDataModule`
* :doc:`Quickstart guide <../_quickstart>`
