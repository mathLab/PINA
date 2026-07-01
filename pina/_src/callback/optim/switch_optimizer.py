"""Module for the SwitchOptimizer callback."""

from lightning.pytorch.callbacks import Callback
from pina._src.optim.optimizer_interface import OptimizerInterface
from pina._src.core.utils import check_consistency, check_positive_integer


class SwitchOptimizer(Callback):
    """
    Lightning callback for dynamically replacing optimizers during training.

    This callback enables switching to one or more new optimizers at a specified
    epoch without restarting the training loop. It is particularly useful for
    staged optimization strategies (e.g., coarse-to-fine training or optimizer
    warm-up phases), where different optimizers are applied sequentially.

    At the target epoch, the provided optimizers are hooked to the model
    parameters and replace the current optimizers in both the PINA solver and
    the Lightning trainer strategy.

    :Example:

        >>> from pina.optim import TorchOptimizer
        >>> import torch
        >>> optimizer = TorchOptimizer(torch.optim.Adam, lr=0.01)
        >>> switch = SwitchOptimizer(new_optimizers=optimizer, epoch_switch=10)
        >>> switch._epoch_switch
        10
    """

    def __init__(self, new_optimizers, epoch_switch):
        """
        Initialization of the :class:`SwitchOptimizer` class.

        :param new_optimizers: The model optimizers to switch to. Can be a
            single :class:`torch.optim.Optimizer` instance or a list of them
            for multiple model solver.
        :type new_optimizers: pina.optim.OptimizerInterface | list
        :param int epoch_switch: The epoch at which the optimizer switch occurs.
        :raises AssertionError: If ``epoch_switch`` is not a positive integer.
        :raises ValueError: If any of the provided optimizers are not instances
            of :class:`pina.optim.OptimizerInterface`.

        Example:
            >>> optimizer = TorchOptimizer(torch.optim.Adam, lr=0.01)
            >>> switch_callback = SwitchOptimizer(
            >>>     new_optimizers=optimizer, epoch_switch=10
            >>> )
        """
        super().__init__()

        # Check consistency
        check_positive_integer(epoch_switch, strict=True)
        check_consistency(new_optimizers, OptimizerInterface)

        # If new_optimizers is not a list, convert it to a list
        if not isinstance(new_optimizers, list):
            new_optimizers = [new_optimizers]

        # Store the new optimizers and epoch switch
        self._new_optimizers = new_optimizers
        self._epoch_switch = epoch_switch

    def on_train_epoch_start(self, trainer, __):
        """
        Switch the optimizer at the start of the specified training epoch.

        :param Trainer trainer: The trainer object managing the training
            process.
        :param __: Placeholder argument, not used.
        """
        # Check if the current epoch matches the switch epoch
        if trainer.current_epoch == self._epoch_switch:
            optims = []

            # Hook the new optimizers to the model parameters
            for idx, optim in enumerate(self._new_optimizers):
                optim.hook(trainer.solver._pina_models[idx].parameters())
                optims.append(optim)

            # Update the solver's optimizers
            trainer.solver._pina_optimizers = optims

            # Update the trainer's strategy optimizers
            trainer.strategy.optimizers = [o.instance for o in optims]
