"""Public API for Trainer.

:Example:

    >>> from pina.trainer import Trainer
    >>> # trainer = Trainer(solver=solver, max_epochs=1000)
"""

from pina._src.core.trainer import Trainer

__all__ = ["Trainer"]
