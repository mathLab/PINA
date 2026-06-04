"""Module for the gradient-enhanced mixin class."""

import torch
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.core.utils import check_consistency
from pina._src.core.operator import grad


class _GradientEnhancedMixin:
    """
    Mixin that augments residual losses with a gradient-based regularization
    term.

    The additional penalty is the norm of the residual gradient with respect
    to the spatial input variables. It is only applied to the conditions whose
    names are listed in ``regularized_conditions``.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver` and using
    :class:`~pina._src.core.tensor.label_tensor.LabelTensor` as input.
    """

    def _init_gradient_enhanced_components(
        self, regularization_weight=1.0, regularized_conditions=None
    ):
        """
        Initialize the gradient-enhancement parameters.

        :param regularization_weight: The weight of the gradient regularization
            term. Default is ``1.0``.
        :type regularization_weight: float | int
        :param regularized_conditions: The names of the conditions that should
            receive gradient regularization. If ``None``, all conditions are
            regularized. Default is ``None``.
        :type regularized_conditions: str | list[str]
        :raises ValueError: If ``regularization_weight`` is not a float or int.
        :raises ValueError: If ``regularized_conditions`` is not a string or a
            list of strings.
        :raises ValueError: If ``problem`` is not an instance of
            :class:`~pina._src.problem.spatial_problem.SpatialProblem`.
        :raises ValueError: If the solver's input data are not instances of
            :class:`~pina._src.core.tensor.label_tensor.LabelTensor`.
        :raises ValueError: If any of the specified ``regularized_conditions``
            are not present in the ``problem``'s conditions.
        """
        # Use all conditions if regularized_conditions is None
        if regularized_conditions is None:
            regularized_conditions = list(self.problem.conditions.keys())

        # Check consistency
        check_consistency(regularization_weight, (float, int))
        check_consistency(regularized_conditions, str)

        # Map conditions to list if a single condition is provided
        if not isinstance(regularized_conditions, (list, tuple)):
            regularized_conditions = [regularized_conditions]

        # Assert the problem is instance of SpatialProblem
        if not isinstance(self.problem, SpatialProblem):
            raise ValueError(
                "Gradient-enhanced regularization requires the problem to be "
                f"an instance of SpatialProblem. Got {type(self.problem)}."
            )

        # Ensure that the solver is using LabelTensors as input
        if not self.use_lt:
            raise ValueError(
                "Gradient-enhanced regularization requires the solver to use "
                f"LabelTensors as input. Got use_lt={self.use_lt}."
            )

        # Ensure that all regularized conditions are present in the problem
        problem_conditions = set(self.problem.conditions.keys())
        for condition in regularized_conditions:
            if condition not in problem_conditions:
                raise ValueError(
                    f"Condition '{condition}' is not present in the problem."
                )

        # Initialize the gradient-enhancement parameters
        self.regularization_weight = regularization_weight
        self.regularized_conditions = regularized_conditions

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

        # If data does not require grad, force requires_grad to True
        if "input" in data and not data["input"].requires_grad:
            data["input"].requires_grad_(True)

        # Compute and store the residual tensor for the condition
        self.residual_tensor = condition.evaluate(data, self)
        self.residual_tensor.labels = [
            f"res_{i}" for i in range(self.residual_tensor.shape[1])
        ]

        # Retrieve condition name for more complex weighting schemes
        condition_name = condition.name if hasattr(condition, "name") else None

        # Compute the tensor loss from the residual tensor
        condition_tensor_loss = self._loss_from_residual(condition_name)

        # Regularize the loss with the gradient penalty if needed
        if condition_name in self.regularized_conditions:

            # Compute the gradient of the residual with respect to spatial input
            residual_gradient = grad(
                output_=self.residual_tensor,
                input_=data["input"],
                d=self.problem.spatial_variables,
            )

            # Compute the norm of the residual gradient
            residual_gradient_norm = self._loss_fn(
                residual_gradient, torch.zeros_like(residual_gradient)
            )

            # Compute the gradient penalty term
            penalty = self.regularization_weight * residual_gradient_norm

            # Add the gradient penalty to the original condition tensor loss
            condition_tensor_loss = condition_tensor_loss + penalty

        # Compute the scalar loss from the tensor loss and return it
        condition_scalar_loss = self._apply_reduction(condition_tensor_loss)

        return condition_scalar_loss
