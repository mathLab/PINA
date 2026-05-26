"""Module for the multi-model solver class."""

from pina._src.solver.mixin.multi_model_mixin import _MultiModelMixin
from pina._src.solver.base_solver import BaseSolver
from pina._src.solver.mixin.manual_optimization_mixin import (
    _ManualOptimizationMixin,
)
from pina._src.solver.mixin.condition_aggregator_mixin import (
    _ConditionAggregatorMixin,
)


class MultiModelSolver(
    _ManualOptimizationMixin,
    _MultiModelMixin,
    _ConditionAggregatorMixin,
    BaseSolver,
):
    """
    Base class for implementing multi-model solvers.

    This class provides the standard starting point for solvers based on
    multiple models. It combines the shared solver machinery from
    :class:`~pina._src.solver.base_solver.BaseSolver` with multi-model handling,
    manual optimization, and condition-wise loss aggregation.

    Subclasses can inherit from this class to implement solver-specific behavior
    while reusing the common logic for model registration, optimizer and
    scheduler setup, manual optimization, loss evaluation, weighting, and
    aggregation across problem conditions.
    """

    def __init__(
        self,
        problem,
        models,
        optimizers=None,
        schedulers=None,
        weighting=None,
        loss=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`MultiModelSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param models: The model or list of models used by the solver.
        :type models: torch.nn.Module | list[torch.nn.Module]
        :param optimizers: The optimizer or list of optimizers used by the
            solver. If ``None``, the ``torch.optim.Adam`` optimizer with a
            learning rate of ``0.001`` is used for each model.
            Default is ``None``.
        :type optimizers: TorchOptimizer | list[TorchOptimizer]
        :param schedulers: The scheduler or list of schedulers used by the
            solver. If ``None``, the ``torch.optim.lr_scheduler.ConstantLR``
            scheduler with a factor of ``1.0`` is used for each model.
            Default is ``None``.
        :type schedulers: TorchScheduler | list[TorchScheduler]
        :param BaseWeighting weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
            Default is ``True``.
        """

        # Initialize the base solver
        BaseSolver.__init__(self, problem=problem, use_lt=use_lt)

        # Initialize the components of the solver
        self._init_solver_components(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
        )

        # Initialize the weighting scheme for the conditions and the loss
        self._init_weighting_and_loss(weighting=weighting, loss=loss)

        # Activate manual optimization
        self._init_manual_optimization()
