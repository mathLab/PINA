"""Module for the MultiModelSimpleSolver."""

import torch
from torch.nn.modules.loss import _Loss

from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.condition.input_equation_condition import (
    InputEquationCondition,
)
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.core.utils import check_consistency
from pina._src.loss.loss_interface import LossInterface
from pina._src.solver.solver import MultiSolverInterface


class MultiModelSimpleSolver(MultiSolverInterface):
    """
    Minimal multi-model solver with explicit residual evaluation, reduction,
    and loss aggregation across conditions.

    The solver orchestrates a uniform workflow for all conditions in the batch.
    Each model in the ensemble contributes its own forward pass independently,
    and the outputs are stacked along ``ensemble_dim``:

    .. math::
        \\hat{\\mathbf{u}}_i = \\mathcal{M}_i(\\mathbf{s}),
        \\quad i = 1, \\dots, N_{\\rm ensemble}

    During the optimization cycle each model's prediction is evaluated against
    the condition independently, and the resulting per-model losses are
    averaged to form the aggregated condition loss:

    .. math::
        \\mathcal{L}_{\\rm condition} = \\frac{1}{N_{\\rm ensemble}}
        \\sum_{i=1}^{N_{\\rm ensemble}} \\mathcal{L}_i

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

    def __init__(
        self,
        problem,
        models,
        optimizers=None,
        schedulers=None,
        weighting=None,
        loss=None,
        use_lt=True,
        ensemble_dim=0,
    ):
        """
        Initialize the multi-model simple solver.

        :param AbstractProblem problem: The problem to be solved.
        :param list[torch.nn.Module] models: The neural network models to be
            used. Must be a list or tuple with at least two models.
        :param list[Optimizer] optimizers: The optimizers to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used for
            each model. Default is ``None``.
        :param list[Scheduler] schedulers: The learning rate schedulers.
            If ``None``, :class:`torch.optim.lr_scheduler.ConstantLR` is used
            for each model. Default is ``None``.
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

        check_consistency(loss, (LossInterface, _Loss), subclass=False)
        check_consistency(ensemble_dim, int)

        super().__init__(
            problem=problem,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            weighting=weighting,
            use_lt=use_lt,
        )

        self._loss_fn = loss
        self._reduction = getattr(loss, "reduction", "mean")
        self._ensemble_dim = ensemble_dim

        if hasattr(self._loss_fn, "reduction"):
            self._loss_fn.reduction = "none"

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
            dim=self._ensemble_dim,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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
                self.forward = (  # noqa: E731
                    lambda x, _idx=idx: self.models[_idx].forward(x)
                )
                loss_tensor = condition.evaluate(
                    condition_data, self, self._loss_fn
                )
                self.forward = original_forward
                per_model_losses.append(self._apply_reduction(loss_tensor))

            condition_losses[condition_name] = torch.stack(
                per_model_losses
            ).mean()

        return condition_losses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_reduction(self, value):
        """
        Apply the configured reduction to a non-aggregated condition tensor.

        :param value: The non-aggregated tensor returned by a condition.
        :type value: torch.Tensor
        :return: The reduced scalar tensor.
        :rtype: torch.Tensor
        :raises ValueError: If the reduction is not supported.
        """
        if self._reduction == "none":
            return value
        if self._reduction == "mean":
            return value.mean()
        if self._reduction == "sum":
            return value.sum()
        raise ValueError(f"Unsupported reduction '{self._reduction}'.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def loss(self):
        """
        The underlying element-wise loss module.

        :return: The stored loss module.
        :rtype: torch.nn.Module
        """
        return self._loss_fn

    @property
    def ensemble_dim(self):
        """
        The dimension along which the per-model outputs are stacked.

        :return: The ensemble dimension.
        :rtype: int
        """
        return self._ensemble_dim

    @property
    def num_models(self):
        """
        The number of models in the ensemble.

        :return: The number of models.
        :rtype: int
        """
        return len(self.models)
