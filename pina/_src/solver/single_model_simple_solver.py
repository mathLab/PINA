"""Module for the SingleModelSimpleSolver."""

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
from pina._src.solver.solver import SingleSolverInterface


class SingleModelSimpleSolver(SingleSolverInterface):
    """
    Minimal single-model solver with explicit residual evaluation, reduction,
    and loss aggregation across conditions.

    The solver orchestrates a uniform workflow for all conditions in the batch:

         1. evaluate the condition and obtain a non-aggregated loss tensor;
         2. apply a reduction to obtain a scalar loss for that condition;
     4. return the per-condition losses, which are aggregated by the inherited
       solver machinery through the configured weighting.
    """

    accepted_conditions_types = (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
        use_lt=True,
    ):
        """
        Initialize the single-model simple solver.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param Optimizer optimizer: The optimizer to be used.
        :param Scheduler scheduler: Learning rate scheduler.
        :param WeightingInterface weighting: The weighting schema to be used.
        :param torch.nn.Module loss: The element-wise loss module whose
            reduction strategy is reused by the solver. If ``None``,
            :class:`torch.nn.MSELoss` is used.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
        """
        if loss is None:
            loss = torch.nn.MSELoss()

        check_consistency(loss, (LossInterface, _Loss), subclass=False)

        super().__init__(
            model=model,
            problem=problem,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )

        self._loss_fn = loss
        self._reduction = getattr(loss, "reduction", "mean")

        if hasattr(self._loss_fn, "reduction"):
            self._loss_fn.reduction = "none"

    def optimization_cycle(self, batch):
        """
        Compute one reduced loss per condition in the batch.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The reduced losses for all conditions.
        :rtype: dict[str, torch.Tensor]
        """
        condition_losses = {}

        for condition_name, data in batch:
            condition = self.problem.conditions[condition_name]
            condition_data = dict(data)

            if hasattr(condition_data.get("input"), "requires_grad_"):
                condition_data["input"] = condition_data[
                    "input"
                ].requires_grad_()

            condition_loss_tensor = condition.evaluate(
                condition_data, self, self._loss_fn
            )
            condition_losses[condition_name] = self._apply_reduction(
                condition_loss_tensor
            )

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
        if self._reduction == "none":
            return value
        if self._reduction == "mean":
            return value.mean()
        if self._reduction == "sum":
            return value.sum()
        raise ValueError(f"Unsupported reduction '{self._reduction}'.")

    @property
    def loss(self):
        """
        The underlying element-wise loss module.

        :return: The stored loss module.
        :rtype: torch.nn.Module
        """
        return self._loss_fn
