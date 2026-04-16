"""Module for the MultiSolverInterface base class."""

from abc import ABCMeta
import torch

from pina._src.optim.optimizer_interface import Optimizer
from pina._src.optim.scheduler_interface import Scheduler
from pina._src.core.utils import check_consistency
from pina._src.solver.solver_interface import SolverInterface


class MultiSolverInterface(SolverInterface, metaclass=ABCMeta):
    """
    Base class for PINA solvers using multiple :class:`torch.nn.Module`.
    """

    def __init__(
        self,
        problem,
        models,
        optimizers=None,
        schedulers=None,
        weighting=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`MultiSolverInterface` class.

        :param AbstractProblem problem: The problem to be solved.
        :param models: The neural network models to be used.
        :type model: list[torch.nn.Module] | tuple[torch.nn.Module]
        :param list[Optimizer] optimizers: The optimizers to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used for
            all models. Default is ``None``.
        :param list[Scheduler] schedulers: The schedulers to be used.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used for all the models. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
        :raises ValueError: If the models are not a list or tuple with length
            greater than one.

        .. warning::
            :class:`MultiSolverInterface` uses manual optimization by setting
            ``automatic_optimization=False`` in
            :class:`~lightning.pytorch.core.LightningModule`. For more
            information on manual optimization please
            see `here <https://lightning.ai/docs/pytorch/stable/\
                model/manual_optimization.html>`_.
        """
        if not isinstance(models, (list, tuple)) or len(models) < 2:
            raise ValueError(
                "models should be list[torch.nn.Module] or "
                "tuple[torch.nn.Module] with len greater than "
                "one."
            )

        if optimizers is None:
            optimizers = [
                self.default_torch_optimizer() for _ in range(len(models))
            ]

        if schedulers is None:
            schedulers = [
                self.default_torch_scheduler() for _ in range(len(models))
            ]

        if any(opt is None for opt in optimizers):
            optimizers = [
                self.default_torch_optimizer() if opt is None else opt
                for opt in optimizers
            ]

        if any(sched is None for sched in schedulers):
            schedulers = [
                self.default_torch_scheduler() if sched is None else sched
                for sched in schedulers
            ]

        super().__init__(problem=problem, use_lt=use_lt, weighting=weighting)

        # check consistency of models argument and encapsulate in list
        check_consistency(models, torch.nn.Module)

        # check scheduler consistency and encapsulate in list
        check_consistency(schedulers, Scheduler)

        # check optimizer consistency and encapsulate in list
        check_consistency(optimizers, Optimizer)

        # check length consistency optimizers
        if len(models) != len(optimizers):
            raise ValueError(
                "You must define one optimizer for each model."
                f"Got {len(models)} models, and {len(optimizers)}"
                " optimizers."
            )
        if len(schedulers) != len(optimizers):
            raise ValueError(
                "You must define one scheduler for each optimizer."
                f"Got {len(schedulers)} schedulers, and {len(optimizers)}"
                " optimizers."
            )

        # initialize the model
        self._pina_models = torch.nn.ModuleList(models)
        self._pina_optimizers = optimizers
        self._pina_schedulers = schedulers

        # Set automatic optimization to False.
        # For more information on manual optimization see:
        # http://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        self.automatic_optimization = False

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This method is called at the end of each training batch and overrides
        the PyTorch Lightning implementation to log checkpoints.

        :param torch.Tensor outputs: The ``model``'s output for the current
            batch.
        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param int batch_idx: The index of the current batch.
        """
        # increase by one the counter of optimization to save loggers
        epoch_loop = self.trainer.fit_loop.epoch_loop
        epoch_loop.manual_optimization.optim_step_progress.total.completed += 1
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def configure_optimizers(self):
        """
        Optimizer configuration for the solver.

        :return: The optimizer and the scheduler
        :rtype: tuple[list[Optimizer], list[Scheduler]]
        """
        for optimizer, scheduler, model in zip(
            self.optimizers, self.schedulers, self.models
        ):
            optimizer.hook(model.parameters())
            scheduler.hook(optimizer)

        return (
            [optimizer.instance for optimizer in self.optimizers],
            [scheduler.instance for scheduler in self.schedulers],
        )

    @property
    def models(self):
        """
        The models used for training.

        :return: The models used for training.
        :rtype: torch.nn.ModuleList
        """
        return self._pina_models

    @property
    def optimizers(self):
        """
        The optimizers used for training.

        :return: The optimizers used for training.
        :rtype: list[Optimizer]
        """
        return self._pina_optimizers

    @property
    def schedulers(self):
        """
        The schedulers used for training.

        :return: The schedulers used for training.
        :rtype: list[Scheduler]
        """
        return self._pina_schedulers
