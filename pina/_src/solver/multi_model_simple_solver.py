"""Module for the MultiModelSimpleSolver."""

import torch
from torch.nn.modules.loss import _Loss
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.core.utils import check_consistency, labelize_forward
from pina._src.optim.optimizer_interface import OptimizerInterface
from pina._src.optim.scheduler_interface import SchedulerInterface
from pina._src.loss.loss_interface import DualLossInterface
from pina._src.solver.base_solver import BaseSolver
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.condition.input_equation_condition import (
    InputEquationCondition,
)


class MultiModelSimpleSolver(BaseSolver):
    r"""
    Minimal multi-model solver with explicit residual evaluation, reduction,
    and loss aggregation across conditions.

    The solver orchestrates a uniform workflow for all conditions in the batch.
    Each model in the ensemble contributes its own forward pass independently,
    and the outputs are stacked along ``ensemble_dim``:

    .. math::
        \hat{\mathbf{u}}_i = \mathcal{M}_i(\mathbf{s}),
        \quad i = 1, \dots, N_{\rm ensemble}

    During the optimization cycle each model's prediction is evaluated against
    the condition independently, and the resulting per-model losses are
    averaged to form the aggregated condition loss:

    .. math::
        \mathcal{L}_{\rm condition} = \frac{1}{N_{\rm ensemble}}
        \sum_{i=1}^{N_{\rm ensemble}} \mathcal{L}_i

    The per-condition workflow is:

         1. evaluate the condition for each model and obtain non-aggregated
            loss tensors;
         2. apply the configured reduction to each per-model tensor;
         3. average the reduced per-model losses into a single scalar for
            the condition;
         4. return the per-condition losses, which are aggregated by the
            inherited solver machinery through the configured weighting.
    """

    accepted_conditions_types = (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )

    _AVAILABLE_REDUCTIONS = {
        "none": lambda x: x,
        "mean": lambda x: x.mean(),
        "sum": lambda x: x.sum(),
    }

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
        Initialize the multi-model simple solver.

        :param BaseProblem problem: The problem to be solved.
        :param list[torch.nn.Module] models: The neural network models to be
            used. Must be a list or tuple with at least two models.
        :param list[OptimizerInterface] optimizers: The optimizers to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used for
            each model. Default is ``None``.
        :param list[SchedulerInterface] schedulers: The learning rate
            schedulers. If ``None`` :class:`torch.optim.lr_scheduler.ConstantLR`
            is used for each model. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param torch.nn.Module loss: The element-wise loss module whose
            reduction strategy is reused by the solver. If ``None``,
            :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
            Default is ``True``.
        :param int ensemble_dim: The dimension along which the per-model
            outputs are stacked in :meth:`forward`. Default is ``0``.
        """
        if loss is None:
            loss = torch.nn.MSELoss()

        check_consistency(loss, (DualLossInterface, _Loss), subclass=False)

        super().__init__(
            problem=problem,
            model=models,
            optimizer=optimizers,
            scheduler=schedulers,
            weighting=weighting,
            use_lt=use_lt,
        )

        self._loss_fn = loss
        self._reduction = getattr(loss, "reduction", "mean")

        if hasattr(self._loss_fn, "reduction"):
            self._loss_fn.reduction = "none"
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

        # check consistency of models argument and encapsulate in list
        check_consistency(models, torch.nn.Module)

        # check scheduler consistency and encapsulate in list
        check_consistency(schedulers, SchedulerInterface)

        # check optimizer consistency and encapsulate in list
        check_consistency(optimizers, OptimizerInterface)

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
        self._loss_fn = loss

        # Set automatic optimization to False.
        # For more information on manual optimization see:
        # http://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        self.automatic_optimization = False

    def forward(self, x, model_idx=None):
        """
        Forward pass through the ensemble models.

        If ``model_idx`` is provided, returns the output of the single model
        at that index. Otherwise stacks the outputs of all models along
        ``ensemble_dim``.

        :param LabelTensor x: The input tensor to the models.
        :param int model_idx: Optional index to select a specific model from
            the ensemble. If ``None`` results for all models are stacked in
            the ``ensemble_dim`` dimension. Default is ``None``.
        :return: The output of the selected model, or the stacked outputs from
            all models.
        :rtype: LabelTensor | torch.Tensor
        """
        if model_idx is not None:
            return self.models[model_idx].forward(x)
        return torch.stack(
            [self.forward(x, idx) for idx in range(self.num_models)],
        )

    def training_step(self, batch):
        """
        Training step for the solver, overridden for manual optimization.

        Performs a forward pass, calculates the loss via
        :meth:`optimization_cycle`, applies manual backward propagation and
        runs the optimization step for each model in the ensemble.

        :param list[tuple[str, dict]] batch: A batch of training data. Each
            element is a tuple containing a condition name and a dictionary of
            points.
        :return: The aggregated loss after the training step.
        :rtype: torch.Tensor
        """
        # zero grad for all optimizers
        for opt in self.optimizers:
            opt.instance.zero_grad()
        # compute condition losses (calls optimization_cycle internally via
        # the parent training_step)
        loss = super().training_step(batch)
        # backpropagate
        self.manual_backward(loss)
        # optimizer + scheduler step for each model
        for opt, sched in zip(self.optimizers, self.schedulers):
            opt.instance.step()
            sched.instance.step()
        return loss

    def optimization_cycle(self, batch):
        """
        Compute one reduced, ensemble-averaged loss per condition in the batch.

        For each condition the method evaluates every model independently and
        averages the resulting scalar losses.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The reduced, ensemble-averaged losses for all conditions.
        :rtype: dict[str, torch.Tensor]
        """
        condition_losses = {}

        for condition_name, data in batch:
            condition = self.problem.conditions[condition_name]
            condition_data = dict(data)

            # Evaluate each model independently and average the losses.
            per_model_losses = []
            for idx in range(self.num_models):
                # Temporarily expose only one model through forward so that
                # condition.evaluate uses just that model.
                original_forward = self.forward
                self.forward = lambda x, _idx=idx: self.models[  # noqa: E731
                    _idx
                ].forward(x)

                problem = self.problem
                self.forward = labelize_forward(
                    self.forward,
                    input_variables=problem.input_variables,
                    output_variables=problem.output_variables,
                )
                loss_tensor = condition.evaluate(
                    condition_data, self, self._loss_fn
                ).tensor
                self.forward = original_forward
                per_model_losses.append(self._apply_reduction(loss_tensor))

            condition_losses[condition_name] = torch.stack(
                per_model_losses
            ).mean()

        return condition_losses

    def _apply_reduction(self, value):
        """
        Apply the configured reduction to a non-aggregated condition tensor.

        :param value: The non-aggregated tensor returned by a condition.
        :type value: torch.Tensor
        :return: The reduced scalar tensor.
        :rtype: torch.Tensor
        :raises ValueError: If the reduction is not supported.
        """
        reduction_fn = self._AVAILABLE_REDUCTIONS.get(
            self._reduction
        )

        if reduction_fn is None:
            raise ValueError(
                f"Unsupported reduction '{self._reduction}'. "
                f"Available options include {self._AVAILABLE_REDUCTIONS.keys()}"
            )

        return reduction_fn(value)
    
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
        :rtype: tuple[list[OptimizerInterface], list[SchedulerInterface]]
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
    def loss(self):
        """
        The underlying element-wise loss module.

        :return: The stored loss module.
        :rtype: torch.nn.Module
        """
        return self._loss_fn

    @property
    def num_models(self):
        """
        The number of models in the ensemble.

        :return: The number of models.
        :rtype: int
        """
        return len(self.models)

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
        :rtype: list[OptimizerInterface]
        """
        return self._pina_optimizers

    @property
    def schedulers(self):
        """
        The schedulers used for training.

        :return: The schedulers used for training.
        :rtype: list[SchedulerInterface]
        """
        return self._pina_schedulers
