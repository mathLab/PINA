"""Module for the residual-based attention mixin class."""

import torch
from pina._src.core.utils import check_consistency
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class ResidualBasedAttentionMixin:
    """
    Mixin that augments the residual loss with an attention mechanism based on
    the residual values.

    The attention weights are computed as a function of the residuals, and they
    are used to weight the contribution of each condition to the overall loss.
    This allows the solver to focus more on conditions with larger residuals,
    potentially improving convergence and accuracy.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def _init_residual_attention_components(
        self, eta=0.001, gamma=0.999, regularized_conditions=None
    ):
        """
        Initialize the residual-based attention parameters.

        :param eta: The learning rate for the residual-based attention weights
            update. Default is ``0.001``.
        :type eta: float | int
        :param float gamma: The decay factor for the residual-based attention
            mechanism. Default is ``0.999``.
        :param regularized_conditions: The names of the conditions that should
            receive attention regularization. If ``None``, all conditions are
            regularized. Default is ``None``.
        :type regularized_conditions: str | list[str]
        :raises ValueError: If ``eta`` is not a positive float or int.
        :raises ValueError: If ``gamma`` is not a float in the range (0, 1).
        :raises ValueError: If ``regularized_conditions`` is not a string or a
            list of strings.
        :raises ValueError: If any of the specified ``regularized_conditions``
            are not present in the ``problem``'s conditions.
        """
        # Use all conditions if regularized_conditions is None
        if regularized_conditions is None:
            regularized_conditions = list(self.problem.conditions.keys())

        # Check consistency
        check_consistency(eta, (float, int))
        check_consistency(gamma, float)
        check_consistency(regularized_conditions, str)

        # Assert gamma is in range (0, 1)
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in range (0, 1)")

        # Assert eta is positive
        if eta <= 0:
            raise ValueError("eta must be positive")

        # Map conditions to list if a single condition is provided
        if not isinstance(regularized_conditions, (list, tuple)):
            regularized_conditions = [regularized_conditions]

        # Ensure that all regularized conditions are present in the problem
        problem_conditions = set(self.problem.conditions.keys())
        for condition in regularized_conditions:
            if condition not in problem_conditions:
                raise ValueError(
                    f"Condition '{condition}' is not present in the problem."
                )

        # Initialize residual-based attention parameters
        self.regularized_conditions = regularized_conditions
        self.gamma = gamma
        self.eta = eta
        self.weight_buffers = {}

        # Iterate over all conditions to initialize the attention weights
        for cond in self.regularized_conditions:

            # Get the condition object
            condition = self.problem.conditions[cond]

            # Determine the number of points in the condition
            if isinstance(condition, DomainEquationCondition):
                n_pts = self.problem._discretised_domains[cond].shape[0]
            else:
                n_pts = condition.data.input.shape[0]

            # Register the attention weights as a buffer in the module
            self.register_buffer(f"weight_{cond}", torch.zeros((n_pts, 1)))
            self.weight_buffers[cond] = f"weight_{cond}"

    def _compute_condition_loss(self, condition, data, batch_idx):
        """
        Compute the scalar loss for a given condition and its data.

        :param BaseCondition condition: The condition for which to compute the
            loss.
        :param dict data: The data corresponding to the condition.
        :param int batch_idx: The index of the current batch.
        :return: The scalar loss for the condition.
        :rtype: torch.Tensor
        """
        # Clone the input tensor if it exists to avoid in-place modifications
        if "input" in data and hasattr(data["input"], "clone"):
            data = dict(data)
            data["input"] = data["input"].clone()

        # Compute and store the residual tensor for the condition
        self.residual_tensor = condition.evaluate(data, self)

        # Retrieve condition name for more complex weighting schemes
        condition_name = condition.name

        # Compute the tensor loss from the residual tensor
        condition_tensor_loss = self._loss_from_residual(condition_name)

        # Apply residual-based attention mechanism if needed
        if condition_name in self.regularized_conditions:

            # Compute the normalized residuals norm for the current condition
            res_abs = torch.linalg.vector_norm(
                self.residual_tensor, ord=2, dim=1, keepdim=True
            )
            res_norm = res_abs / (res_abs.max() + 1e-12)

            # Get the correct indices to retrieve the weights for the batch
            len_residuals = self.residual_tensor.shape[0]

            # Get the weights buffer for the current condition
            weights = getattr(self, self.weight_buffers[condition_name])

            # Get the total number of points, together with the start / end idx
            total_points = weights.shape[0]
            start = (batch_idx * len_residuals) % total_points
            end = start + len_residuals

            # Retrieve the weights for the current batch using modular indexing
            idx = torch.arange(start, end, device=weights.device)
            idx = idx % total_points

            # Update weights
            with torch.no_grad():
                weights[idx] = self.gamma * weights[idx] + self.eta * res_norm

            # Weight the condition tensor loss with attention weights
            condition_tensor_loss = condition_tensor_loss * weights[idx]

        # Compute the scalar loss from the tensor loss and return it
        condition_scalar_loss = self._apply_reduction(condition_tensor_loss)

        return condition_scalar_loss
