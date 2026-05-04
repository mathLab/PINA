"""Module for the SwitchScheduler callback."""

from lightning.pytorch.callbacks import Callback
from pina._src.optim.scheduler_interface import SchedulerInterface
from pina._src.core.utils import check_consistency, check_positive_integer


class SwitchScheduler(Callback):
    """
    Lightning callback for dynamically replacing schedulers during training.

    This callback enables switching to new scheduler(s) at a specified epoch
    without interrupting the training loop. It is useful for staged training
    strategies where different learning rate policies are applied sequentially.
    """

    def __init__(self, new_schedulers, epoch_switch):
        """
        Initialization of the :class:`SwitchScheduler` class.

        :param new_schedulers: The scheduler or list of schedulers to switch to.
            Use a single scheduler for single-model solvers, or a list of
            schedulers when working with multiple models.
        :type new_schedulers: SchedulerInterface | list[SchedulerInterface]
        :param int epoch_switch: The epoch at which the scheduler switch occurs.
        :raises AssertionError: If ``epoch_switch`` is not a positive integer.
        :raises ValueError:  If any of the provided schedulers are not instances
            of :class:`pina.optim.SchedulerInterface`.

        Example:
            >>> scheduler = TorchScheduler(
            >>>     torch.optim.lr_scheduler.StepLR, step_size=5
            >>> )
            >>> switch_callback = SwitchScheduler(
            >>>     new_schedulers=scheduler, epoch_switch=10
            >>> )
        """
        super().__init__()

        # Check consistency
        check_positive_integer(epoch_switch, strict=True)
        check_consistency(new_schedulers, SchedulerInterface)

        # If new_schedulers is not a list, convert it to a list
        if not isinstance(new_schedulers, list):
            new_schedulers = [new_schedulers]

        # Store the new schedulers and epoch switch
        self._new_schedulers = new_schedulers
        self._epoch_switch = epoch_switch

    def on_train_epoch_start(self, trainer, __):
        """
        Switch the scheduler at the start of the specified training epoch.

        :param Trainer trainer: The trainer object managing
            the training process.
        :param __: Placeholder argument, not used.
        """
        # Check if the current epoch matches the switch epoch
        if trainer.current_epoch == self._epoch_switch:
            schedulers = []

            # Hook the new schedulers to the model parameters
            for idx, scheduler in enumerate(self._new_schedulers):
                scheduler.hook(trainer.solver._pina_optimizers[idx])
                schedulers.append(scheduler)

                # Update the trainer's scheduler configs
                trainer.lr_scheduler_configs[idx].scheduler = scheduler.instance

            # Update the solver's schedulers
            trainer.solver._pina_schedulers = schedulers
