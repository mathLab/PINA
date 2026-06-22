"""Module for the ensemble mixin class."""

import torch
from pina._src.solver.base_solver import BaseSolver
from pina._src.solver.mixin.multi_model_mixin import MultiModelMixin


class EnsembleMixin(MultiModelMixin):
    """
    Mixin that defines the forward pass and optimizer configuration for solvers
    backed by an ensemble of models. Provides properties to access the models,
    optimizers, and schedulers.

    Designed to be used in combination with any solver inheriting from
    :class:`~pina._src.solver.base_solver.BaseSolver`.
    """

    def forward(self, x):
        """
        Forward pass for ensemble solvers. If an active model index is set, only
        that model is evaluated. Otherwise, all models are evaluated and their
        outputs are stacked together.

        :param x: The input data.
        :type x: torch.Tensor | LabelTensor | Data | Graph
        :return: The output of all models stacked together.
        :rtype: torch.Tensor | LabelTensor | Data | Graph
        """
        # Retrieve the index of the active model if set
        active_idx = getattr(self, "_active_model_idx", None)

        # If an active model index is set, evaluate only that model
        if active_idx is not None:
            return self.models[active_idx](x)

        # Otherwise, evaluate all models and stack outputs
        return torch.stack(
            [self.models[idx](x) for idx in range(self.num_models)]
        )

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
        # Initialize model losses for the current condition
        model_losses = []

        # Restore the active model index if it was set, else set it to None
        previous_active_model_idx = getattr(self, "_active_model_idx", None)

        # Try - finally to ensure active model index is always restored
        try:

            # Iterate over all ensemble models to compute individual losses
            for model_idx in range(self.num_models):

                # Set the active model index for the current iteration
                self._active_model_idx = model_idx

                # Compute the scalar loss for the current model and condition
                condition_scalar_loss = BaseSolver._compute_condition_loss(
                    self, condition, data, batch_idx
                )

                # Store the computed loss for the current model
                model_losses.append(condition_scalar_loss)

        # Ensure that the active model index is always restored
        finally:

            # Restore the previous active model index after computation
            self._active_model_idx = previous_active_model_idx

        return torch.stack(model_losses).mean()
