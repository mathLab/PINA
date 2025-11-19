"""Module for the PINA Scheduler."""

from .optim_connector_interface import SchedulerConnectorInterface, _HooksOptim
from .optimizer_connector import OptimizerConnector
from ...utils import check_consistency


class SchedulerConnector(SchedulerConnectorInterface, _HooksOptim):
    """
    Class for defining a scheduler connector. All specific schedulers connectors
    should inherit form this class and implement the required methods.
    """

    def __init__(self, scheduler_class, **scheduler_kwargs):
        """
        Initialize connector parameters

        :param torch.optim.lr_scheduler.LRScheduler scheduler_class: The torch
            scheduler class.
        :param dict scheduler_kwargs: The scheduler kwargs.
        """
        super().__init__()
        self._scheduler_class = scheduler_class
        self._scheduler_instance = None
        self._scheduler_kwargs = scheduler_kwargs

    def optimizer_hook(self, optimizer):
        """
        Abstract method to define the hook logic for the scheduler. This hook
        is used to hook the scheduler instance with the given optimizer.

        :param Optimizer optimizer: The optimizer to hook.
        """
        check_consistency(optimizer, OptimizerConnector)
        if not optimizer.hooks_done["parameter_hook"]:
            raise RuntimeError(
                "Scheduler cannot be set, Optimizer not hooked "
                "to model parameters. "
                "Please call Optimizer.parameter_hook()."
            )
        self._scheduler_instance = self._scheduler_class(
            optimizer.instance, **self._scheduler_kwargs
        )

    def _register_hooks(self, **kwargs):
        """
        Register the optimizers hooks. This method inspects keyword arguments
        for known keys (`parameters`, `solver`, ...) and applies the
        corresponding hooks.

        It allows flexible integration with
        different workflows without enforcing a strict method signature.

        This method is used inside the
        :class:`~pina.solver.solver.SolverInterface` class.

        :param kwargs: Expected keys may include:
            - ``parameters``: Parameters to be registered for optimization.
            - ``solver``: Solver instance.
        """
        # optimizer hook
        optimizer = kwargs.get("optimizer", None)
        if optimizer is not None:
            check_consistency(optimizer, OptimizerConnector)
            self.optimizer_hook(optimizer)

    @property
    def instance(self):
        """
        Get the scheduler instance.

        :return: The scheduler instance
        :rtype: torch.optim.lr_scheduler.LRScheduler
        """
        return self._scheduler_instance
